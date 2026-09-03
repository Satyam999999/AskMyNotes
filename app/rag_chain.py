"""RAG chain utilities for AskMyNotes — Production Grade.

Improvements over v1:
1. Hybrid Search: dense (BGE) + sparse (SPLADE) vectors fused via RRF
2. Cross-Encoder Re-Ranking via FlashRank (local ONNX, no API cost)
3. Streaming support via `stream_answer()` (SSE-compatible generator)
4. Singleton model caching for dense + sparse embed models
5. Confidence threshold guard preserved
"""

from typing import List, Dict, Generator
import os

# Fix for macOS huggingface tokenizer mutex deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import logging
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Query,
    OrderValue,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import Client as LangSmithClient

logger = logging.getLogger(__name__)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"  # FlashRank local cross-encoder

_dense_model = None
_sparse_model = None
_reranker = None
_global_qdrant_client = None


# ─────────────────────────────────────────────
# Singleton model accessors
# ─────────────────────────────────────────────

def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding
        _cache_dir = os.path.expanduser("~/.cache/fastembed")
        os.makedirs(_cache_dir, exist_ok=True)
        _dense_model = TextEmbedding(DENSE_MODEL, cache_dir=_cache_dir)
    return _dense_model


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        try:
            from fastembed import SparseTextEmbedding
            _cache_dir = os.path.expanduser("~/.cache/fastembed")
            os.makedirs(_cache_dir, exist_ok=True)
            _sparse_model = SparseTextEmbedding(SPARSE_MODEL, cache_dir=_cache_dir)
        except Exception as exc:
            logger.warning("Could not initialize sparse Splade model (%s). Sparse search is disabled.", exc)
            return None
    return _sparse_model


def _get_reranker():
    """FlashRank cross-encoder — local ONNX, no API calls needed."""
    global _reranker
    if _reranker is None:
        from flashrank import Ranker
        _reranker_cache = os.path.expanduser("~/.cache/flashrank")
        os.makedirs(_reranker_cache, exist_ok=True)
        _reranker = Ranker(model_name=RERANKER_MODEL, cache_dir=_reranker_cache)
    return _reranker


def _get_qdrant_client() -> QdrantClient:
    global _global_qdrant_client
    if _global_qdrant_client is not None:
        return _global_qdrant_client

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in environment")

    # QdrantClient's url= param defaults to port 6333 for gRPC even over HTTPS.
    # For Qdrant Cloud (port 443), we must extract the host and set https=True explicitly.
    from urllib.parse import urlparse
    parsed = urlparse(url)
    use_https = parsed.scheme == "https"
    host = parsed.hostname
    port = parsed.port or (443 if use_https else 80)

    _global_qdrant_client = QdrantClient(
        host=host,
        port=port,
        https=use_https,
        api_key=api_key,
        timeout=30.0,
        prefer_grpc=False,
        check_compatibility=False,
    )
    return _global_qdrant_client


def _get_llm() -> ChatGroq:
    model_name = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
    return ChatGroq(model=model_name, temperature=0)


# ─────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────

def embed_dense(text: str) -> List[float]:
    """Embed a single query string with BGE-small."""
    model = _get_dense_model()
    return list(next(model.embed([text])))


def embed_sparse(text: str) -> SparseVector | None:
    """Embed a single query string with SPLADE sparse model."""
    model = _get_sparse_model()
    if model is None:
        return None
    emb = next(model.embed([text]))
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


# ─────────────────────────────────────────────
# Hybrid Retrieval + Re-ranking
# ─────────────────────────────────────────────

def retrieve_chunks(
    question: str,
    collection_name: str,
    top_k: int = 8,
    rerank_top_n: int = 5,
    client: QdrantClient | None = None,
):
    """
    Two-stage retrieval:
    1. Hybrid search (dense BGE + sparse SPLADE fused via RRF) → top_k=25 candidates
    2. Cross-encoder re-ranking (FlashRank ms-marco) → final top_n=rerank_top_n

    Falls back to dense-only search if the collection doesn't have sparse vectors.

    Returns: (results, sources, joined_context)
    """
    client = client or _get_qdrant_client()

    # Check if this collection has sparse vectors configured
    try:
        coll_info = client.get_collection(collection_name)
        has_sparse = bool(getattr(coll_info.config.params, "sparse_vectors", None))
    except Exception:
        has_sparse = False

    # Over-fetch candidates for re-ranking
    fetch_k = max(top_k, 15)

    if has_sparse and _get_sparse_model() is not None:
        # ── HYBRID SEARCH (Improvement 2) ──────────────────────────────
        logger.info("Using hybrid search (dense + sparse) for collection '%s'", collection_name)
        dense_vec = embed_dense(question)
        sparse_vec = embed_sparse(question)

        if sparse_vec is not None:
            try:
                results = client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        Prefetch(query=dense_vec, using="dense", limit=fetch_k),
                        Prefetch(query=sparse_vec, using="sparse", limit=fetch_k),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=fetch_k,
                    with_payload=True,
                ).points
            except Exception as exc:
                logger.warning("Hybrid search failed (%s), falling back to dense-only", exc)
                results = _dense_only_search(client, collection_name, question, fetch_k)
        else:
            results = _dense_only_search(client, collection_name, question, fetch_k)
    else:
        # ── DENSE-ONLY (legacy collections or if sparse model failed to load) ──────
        logger.info("Using dense-only search for collection '%s'", collection_name)
        results = _dense_only_search(client, collection_name, question, fetch_k)

    if not results:
        return [], [], ""

    # ── CROSS-ENCODER RE-RANKING (Improvement 3) ───────────────────────
    results = _rerank(question, results, top_n=rerank_top_n)

    # Build sources and context string
    sources = []
    contexts = []
    for r in results:
        payload = r.payload or {}
        text = payload.get("text") or payload.get("content") or ""
        page = payload.get("page_number")
        sources.append({
            "text": text,
            "page_number": int(page) if page is not None else None,
            "source_file": payload.get("source_file"),
            "section": payload.get("section", ""),
        })
        label = f"[Page {page}]" if page is not None else ""
        contexts.append(f"{label} {text}".strip())

    joined_context = "\n\n---\n\n".join(contexts)
    return results, sources, joined_context


def _dense_only_search(client, collection_name, question, fetch_k):
    """Dense cosine vector search — works with both old and new collections."""
    dense_vec = embed_dense(question)
    if hasattr(client, "query_points"):
        try:
            return client.query_points(
                collection_name=collection_name,
                query=dense_vec,
                using="dense",
                limit=fetch_k,
                with_payload=True,
            ).points
        except Exception:
            pass
    # Fallback to legacy .search()
    return client.search(
        collection_name=collection_name,
        query_vector=dense_vec,
        limit=fetch_k,
        with_payload=True,
    )


def _rerank(question: str, results, top_n: int):
    """
    Cross-encoder re-ranking using FlashRank (ms-marco-MiniLM-L-12-v2).
    Over-fetched candidates are scored against the question and the
    best top_n are returned.
    """
    try:
        from flashrank import RerankRequest
        ranker = _get_reranker()
        passages = [
            {"id": i, "text": (r.payload or {}).get("text", "")}
            for i, r in enumerate(results)
        ]
        rerank_request = RerankRequest(query=question, passages=passages)
        reranked = ranker.rerank(rerank_request)
        # Reorder results by reranker score
        id_to_result = {i: r for i, r in enumerate(results)}
        reranked_results = []
        for item in reranked[:top_n]:
            orig_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if orig_id is not None and orig_id in id_to_result:
                reranked_results.append(id_to_result[orig_id])
        logger.info("Re-ranked %d → %d chunks via FlashRank", len(results), len(reranked_results))
        return reranked_results if reranked_results else results[:top_n]
    except Exception as exc:
        logger.warning("Re-ranking failed (%s), using retrieval order", exc)
        return results[:top_n]


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

BASE_SYSTEM_PROMPT = (
    "You are a multilingual study assistant that gives clear, well-structured answers.\n"
    "Format your answer using these rules:\n"
    "- Start with a 1-2 sentence direct answer\n"
    "- Then provide detail in bullet points (use - for each bullet)\n"
    "- Bold important terms using **term**\n"
    "- Cite the page number after every factual claim like [Page 4]\n"
    "- End with a 'Key Takeaway:' line (translated into the response language)\n"
    "Use ONLY the provided context. If the answer is not found, say so in the response language.\n"
    "IMPORTANT: You MUST respond entirely in {language}. Translate all formatting labels too."
)


def _build_system_prompt(language: str = "English") -> str:
    return BASE_SYSTEM_PROMPT.format(language=language)


def ask_question(question: str, collection_name: str, top_k: int = 8, response_language: str = "English") -> Dict:
    """
    Full RAG pipeline:
    1. Hybrid retrieval (dense + sparse) with over-fetching
    2. Cross-encoder re-ranking to best 5 chunks
    3. LLM generation with citation prompt

    Returns: {answer, sources, model_used, processing_time_ms}
    """
    start = time.time()
    client = _get_qdrant_client()

    # Verify collection exists
    collections = client.get_collections()
    if not any(c.name == collection_name for c in collections.collections):
        raise ValueError(f"Collection '{collection_name}' not found in Qdrant")

    results, sources, joined_context = retrieve_chunks(
        question, collection_name, top_k=top_k, rerank_top_n=3, client=client
    )

    # Confidence guard
    try:
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    except Exception:
        threshold = 0.3

    max_score = max(
        (getattr(r, "score", 0) or 0 for r in results), default=None
    )
    if not results or (max_score is not None and max_score < threshold):
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "answer": "Not found in your notes.",
            "sources": [],
            "model_used": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "processing_time_ms": elapsed_ms,
        }

    model_used = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
    llm = ChatGroq(model=model_used, temperature=0)
    system_prompt = _build_system_prompt(response_language)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{joined_context}\n\nQuestion: {question}\n\nAnswer (respond in {response_language}):"),
    ]
    answer_text = (llm | StrOutputParser()).invoke(messages)

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "answer": answer_text,
        "sources": sources,
        "model_used": model_used,
        "processing_time_ms": elapsed_ms,
    }


def stream_answer(question: str, collection_name: str, top_k: int = 8, response_language: str = "English") -> Generator[str, None, None]:
    """
    Streaming version of ask_question (Improvement 5 — SSE support).
    Retrieves and re-ranks chunks, then streams LLM tokens as a generator.
    Use with FastAPI StreamingResponse / Server-Sent Events.
    """
    client = _get_qdrant_client()
    _, _, joined_context = retrieve_chunks(
        question, collection_name, top_k=top_k, rerank_top_n=3, client=client
    )

    model_used = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
    llm = ChatGroq(model=model_used, temperature=0, streaming=True)
    system_prompt = _build_system_prompt(response_language)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{joined_context}\n\nQuestion: {question}\n\nAnswer (respond in {response_language}):"),
    ]
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", "")
        if content:
            yield content


# ─────────────────────────────────────────────
# LangSmith helpers (preserved from v1)
# ─────────────────────────────────────────────

def _langsmith_project_name() -> str:
    return os.getenv("LANGCHAIN_PROJECT", "askmynotes")


def was_last_run_traced_successfully(run_name: str, question: str, collection_name: str, limit: int = 10) -> bool:
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true" or not os.getenv("LANGCHAIN_API_KEY"):
        return False
    try:
        client = LangSmithClient()
        project_name = _langsmith_project_name()
        for run in client.list_runs(project_name=project_name, run_type="chain", is_root=True, limit=limit):
            if getattr(run, "name", None) != run_name:
                continue
            inputs = getattr(run, "inputs", {}) or {}
            if inputs.get("question") == question and inputs.get("collection_name") == collection_name:
                return True
        return False
    except Exception as exc:
        logger.warning("Could not verify LangSmith trace for %s: %s", run_name, exc)
        return False
