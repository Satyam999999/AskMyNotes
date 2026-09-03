from __future__ import annotations

"""
Ingestion pipeline for AskMyNotes RAG system — Production Grade.

Improvements over v1:
1. Semantic Markdown Parsing via pymupdf4llm (preserves tables, headers, layout)
2. Sparse embeddings stored alongside dense vectors (enables Hybrid Search)
3. UUID-based point IDs (safe multi-PDF uploads without overwrites)
4. Larger chunks (800 / 100 overlap) for richer contextual paragraphs
5. Named vectors (dense + sparse) stored in Qdrant for hybrid retrieval
"""

import os
import sys

# Fix for macOS huggingface tokenizer mutex deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import uuid
from pathlib import Path
import logging
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
VECTOR_SIZE = 384

_dense_model = None
_sparse_model = None


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
            logger.warning("Could not initialize sparse Splade model (%s). Sparse indexing is disabled.", exc)
            return None
    return _sparse_model


def embed_dense(texts: List[str]) -> List[List[float]]:
    """Generate dense BGE vectors for a batch of texts."""
    model = _get_dense_model()
    return [list(emb) for emb in model.embed(texts)]


def embed_sparse(texts: List[str]) -> List[SparseVector] | None:
    """Generate SPLADE sparse vectors for a batch of texts."""
    from qdrant_client.models import SparseVector

    model = _get_sparse_model()
    if model is None:
        return None
    results = []
    for emb in model.embed(texts):
        results.append(SparseVector(
            indices=list(emb.indices.tolist()),
            values=list(emb.values.tolist()),
        ))
    return results


def get_qdrant_client() -> QdrantClient:
    """Initialize and return a cached Qdrant Cloud client."""
    from qdrant_client import QdrantClient
    from urllib.parse import urlparse

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in .env file."
        )

    parsed = urlparse(qdrant_url)
    use_https = parsed.scheme == "https"
    host = parsed.hostname
    port = parsed.port or (443 if use_https else 80)

    client = QdrantClient(
        host=host,
        port=port,
        https=use_https,
        api_key=qdrant_api_key,
        prefer_grpc=False,
        timeout=30.0,
        check_compatibility=False,
    )
    client.get_collections()  # Verify connection
    logger.info("✓ Connected to Qdrant Cloud: %s", qdrant_url)
    return client


def _ensure_collection(client: QdrantClient, collection_name: str, has_sparse: bool = True) -> None:
    """Create or verify a Qdrant collection that supports dense (and optionally sparse) vectors."""
    from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

    collections = client.get_collections()
    exists = any(c.name == collection_name for c in collections.collections)

    if not exists:
        logger.info("📦 Creating collection: %s", collection_name)
        vectors_config = {
            "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        }
        sparse_config = {
            "sparse": SparseVectorParams(index=SparseIndexParams()),
        } if has_sparse else None

        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
        logger.info("✓ Collection created")
    else:
        logger.info("✓ Collection '%s' already exists", collection_name)


def _parse_pdf_to_pages(pdf_path: Path) -> List[Dict]:
    """
    IMPROVEMENT 1: Semantic Markdown Parsing (Page-by-Page).
    Uses pymupdf4llm to convert PDF to Markdown page-by-page, preserving tables,
    headings, and exact page numbers. This is vastly superior to raw
    text extraction (PyMuPDFLoader).
    """
    import pymupdf4llm
    logger.info("📄 Parsing PDF to Page-by-Page Markdown: %s", pdf_path)
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    logger.info("✓ Converted %d pages to Markdown", len(pages))
    return pages


def _split_markdown_pages(pages: List[Dict], source_file: str):
    """
    Split Markdown by headers first, then recursively by size, preserving page numbers.
    Chunks respect paragraph and section boundaries rather than arbitrary character limits.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    # Primary split: by Markdown headers (semantic boundaries)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    # Secondary split: ensure no chunk exceeds 800 chars
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page in pages:
        page_num = page.get("metadata", {}).get("page_number")
        if page_num is None:
            page_num = page.get("metadata", {}).get("page")
        
        page_text = page.get("text", "")
        if not page_text.strip():
            continue

        header_docs = header_splitter.split_text(page_text)
        for doc in header_docs:
            sub_chunks = char_splitter.split_text(doc.page_content)
            for text in sub_chunks:
                if text.strip():
                    chunks.append({
                        "text": text.strip(),
                        "source_file": source_file,
                        "section": doc.metadata.get("h1") or doc.metadata.get("h2") or doc.metadata.get("h3") or "",
                        "page_number": page_num,
                    })

    return chunks


def ingest_pdf(pdf_path: str, collection_name: str, original_filename: str | None = None) -> Dict:
    """
    Full ingestion pipeline with:
    - Page-by-page Semantic Markdown parsing (Point 1)
    - Dense + Sparse vectors stored per chunk (Point 2)
    - UUID point IDs for safe multi-PDF upload (Point 3)

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        collection_name: Qdrant collection name.
        original_filename: Optional original user-friendly filename.

    Returns:
        dict with chunks_ingested, collection_name, source_file counts.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    # --- Step 1: Semantic Markdown Parsing ---
    pages = _parse_pdf_to_pages(pdf_path)

    # --- Step 2: Semantic Chunking ---
    logger.info("📑 Splitting into semantic chunks...")
    filename = original_filename or pdf_path.name
    chunks = _split_markdown_pages(pages, filename)
    logger.info("✓ Created %d chunks", len(chunks))

    if not chunks:
        raise RuntimeError("No content extracted from PDF. File may be image-only.")

    # --- Step 3: Embed (Dense + Sparse) ---
    logger.info("🧠 Generating embeddings...")
    texts = [c["text"] for c in chunks]
    dense_vecs = embed_dense(texts)
    sparse_vecs = embed_sparse(texts)
    has_sparse = sparse_vecs is not None
    if has_sparse:
        logger.info("✓ Embedded %d chunks (dense + sparse)", len(chunks))
    else:
        logger.info("✓ Embedded %d chunks (dense only)", len(chunks))

    # --- Step 4: Connect and ensure collection ---
    client = get_qdrant_client()
    _ensure_collection(client, collection_name, has_sparse=has_sparse)

    # --- Step 5: Upsert with UUID IDs ---
    logger.info("💾 Uploading vectors to Qdrant...")
    from qdrant_client.models import PointStruct

    points = []
    
    if has_sparse:
        for chunk, d_vec, s_vec in zip(chunks, dense_vecs, sparse_vecs):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d_vec,
                        "sparse": s_vec,
                    },
                    payload={
                        "text": chunk["text"],
                        "source_file": chunk["source_file"],
                        "section": chunk["section"],
                        "page_number": chunk["page_number"],
                    },
                )
            )
    else:
        for chunk, d_vec in zip(chunks, dense_vecs):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d_vec,
                    },
                    payload={
                        "text": chunk["text"],
                        "source_file": chunk["source_file"],
                        "section": chunk["section"],
                        "page_number": chunk["page_number"],
                    },
                )
            )

    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "  Uploaded batch %d/%d",
            i // batch_size + 1,
            (len(points) + batch_size - 1) // batch_size,
        )

    logger.info("✓ Successfully ingested %d chunks into '%s'", len(points), collection_name)
    return {
        "chunks_ingested": len(points),
        "collection_name": collection_name,
        "source_file": pdf_path.name,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <pdf_path> <collection_name>")
        sys.exit(1)

    result = ingest_pdf(sys.argv[1], sys.argv[2])
    print(f"\n✅ Ingestion successful!")
    print(f"   Chunks ingested: {result['chunks_ingested']}")
    print(f"   Collection: {result['collection_name']}")
    print(f"   Source file: {result['source_file']}")
