"""FastAPI application for AskMyNotes.

Endpoints:
- POST /upload: upload PDF, run ingestion, save metadata to SQLite
- POST /ask: ask a question against a collection
- GET /health: basic health + Qdrant connectivity
- POST /revision: study mode revision sheet
- POST /quiz: study mode MCQs
- POST /explain-simple: simplified explanation
- POST /audio-notes: MP3 revision notes
- POST /night-before: condensed cheat sheet
- POST /flashcards: flashcards
- POST /highlights: sentence highlights
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from importlib import import_module

# Fix for macOS huggingface tokenizer mutex deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Fix for macOS gRPC and multiprocessing fork deadlocks
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

APP_DB = os.path.join(os.path.dirname(__file__), "metadata.db")


def _ensure_db() -> None:
    conn = sqlite3.connect(APP_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            collection_id TEXT PRIMARY KEY,
            filename TEXT,
            chunk_count INTEGER,
            upload_ts TEXT
        )
        """
    )
    conn.commit()
    conn.close()


class UploadResponse(BaseModel):
    collection_id: str
    chunks_created: int


SUPPORTED_LANGUAGES = {
    "en": "English",
    "ja": "Japanese (日本語)",
    "hi": "Hindi (हिंदी)",
    "zh": "Chinese (中文)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ko": "Korean (한국어)",
    "ar": "Arabic (العربية)",
    "pt": "Portuguese (Português)",
}


class AskRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None
    language: str = "en"  # ISO 639-1 language code


class TopicRequest(BaseModel):
    topic: str
    collection_id: Optional[str] = None


class ConceptRequest(BaseModel):
    concept: str
    collection_id: Optional[str] = None


class NightBeforeRequest(BaseModel):
    subject: str
    collection_id: Optional[str] = None
    exam_hours_away: int = 8


class QuizRequest(BaseModel):
    topic: str
    collection_id: Optional[str] = None
    num_questions: int = 5


class FlashcardsRequest(BaseModel):
    topic: str
    collection_id: Optional[str] = None
    num_cards: int = 10


class HighlightsRequest(BaseModel):
    topic: str
    collection_id: Optional[str] = None


class SourceItem(BaseModel):
    text: str
    page_number: Optional[int] = None
    source_file: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool


class RevisionResponse(BaseModel):
    revision_sheet: str
    sources: List[SourceItem]
    processing_time_ms: int


class SimpleExplanationResponse(BaseModel):
    simple_explanation: str
    analogy: str
    one_thing_to_remember: str
    sources: List[SourceItem]
    processing_time_ms: int


class NightBeforeResponse(BaseModel):
    cheat_sheet: str
    topics_covered: List[str]
    sources: List[SourceItem]
    processing_time_ms: int


class QuizItem(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct: str
    explanation: str
    source_page: Optional[int] = None


class QuizResponse(BaseModel):
    quiz: List[QuizItem]
    processing_time_ms: int


class FlashcardItem(BaseModel):
    front: str
    back: str
    difficulty: str
    source_page: Optional[int] = None


class FlashcardsResponse(BaseModel):
    flashcards: List[FlashcardItem]
    processing_time_ms: int


class HighlightItem(BaseModel):
    sentence: str
    score: int
    reason: str
    page_number: Optional[int] = None


class HighlightsResponse(BaseModel):
    highlights: List[HighlightItem]
    processing_time_ms: int


app = FastAPI(title="AskMyNotes")


def _get_allowed_origins() -> List[str]:
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://ask-my-notes-ruby.vercel.app",
    ]
    extra = os.getenv("CORS_ALLOW_ORIGINS", "")
    parsed_extra = [origin.strip().rstrip("/") for origin in extra.split(",") if origin.strip()]

    # Keep order stable while removing duplicates.
    deduped: List[str] = []
    for origin in defaults + parsed_extra:
        if origin not in deduped:
            deduped.append(origin)
    return deduped


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    _ensure_db()
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY is missing; LangSmith tracing will not be available.")
    elif os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        logger.warning("LANGCHAIN_TRACING_V2 is not enabled; LangSmith traces will not be recorded.")
    else:
        logger.info("LangSmith tracing is enabled for project %s", os.getenv("LANGCHAIN_PROJECT", "askmynotes"))


GLOBAL_COLLECTION = "askmynotes_v2"


@lru_cache(maxsize=1)
def _get_rag_chain_module():
    # Import the RAG stack lazily so Cloud Run can bind to PORT quickly.
    return import_module("app.rag_chain")


@lru_cache(maxsize=1)
def _get_ingest_pdf_func():
    # The ingestion pipeline pulls in heavy parsing/embedding deps; keep it off the hot path.
    return import_module("app.ingest").ingest_pdf

def _collection_exists() -> None:
    try:
        rag_chain = _get_rag_chain_module()
        client = rag_chain._get_qdrant_client()
        collections = client.get_collections()
        if not any(c.name == GLOBAL_COLLECTION for c in collections.collections):
            pass # It will be created on upload
    except Exception as exc:
        logger.exception("Qdrant collection check failed")
        raise HTTPException(status_code=503, detail=str(exc))


def _get_llm() -> ChatGroq:
    model_name = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
    return ChatGroq(model=model_name, temperature=0)


def _build_context_bundle(sources: List[dict]) -> str:
    parts = []
    for source in sources:
        text = source.get("text") or ""
        page = source.get("page_number")
        label = f"[Page {page}]" if page is not None else ""
        parts.append(f"{label} {text}".strip())
    return "\n\n---\n\n".join(parts)


def _retrieve_sources(query: str, top_k: int = 8):
    rag_chain = _get_rag_chain_module()
    _, sources, joined_context = rag_chain.retrieve_chunks(query, GLOBAL_COLLECTION, top_k=top_k)
    return sources, joined_context


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _invoke_json_llm(prompt: str, system: str, retries: int = 2) -> dict | list:
    # Use direct message objects to avoid LangChain treating JSON shape examples
    # like {"key"} as template variables, which causes INVALID_PROMPT_INPUT errors.
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        llm = _get_llm().bind(response_format={"type": "json_object"})
    except Exception:
        llm = _get_llm()

    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        raw = (llm | StrOutputParser()).invoke(messages)
        try:
            return json.loads(_strip_json_fences(raw))
        except Exception as exc:
            last_error = exc
            logger.warning("JSON parsing failed on attempt %s/%s: %s | Raw output: %r", attempt + 1, retries + 1, exc, raw)
    raise ValueError(f"Model did not return valid JSON: {last_error}")


def _ensure_list_of_strings(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


@app.post("/upload", response_model=UploadResponse)
async def upload(pdf: UploadFile = File(...)):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    max_bytes = 10 * 1024 * 1024
    temp_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{temp_id}.pdf"
    written = 0
    try:
        ingest_pdf = _get_ingest_pdf_func()
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await pdf.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="File too large (max 10MB)")
                out.write(chunk)

        result = ingest_pdf(tmp_path, GLOBAL_COLLECTION, original_filename=pdf.filename)
        chunks = int(result.get("chunks_ingested", 0))

        conn = sqlite3.connect(APP_DB)
        cur = conn.cursor()
        cur.execute(
            "REPLACE INTO uploads (collection_id, filename, chunk_count, upload_ts) VALUES (?, ?, ?, datetime('now'))",
            (GLOBAL_COLLECTION, pdf.filename, chunks),
        )
        conn.commit()
        conn.close()

        return {"collection_id": GLOBAL_COLLECTION, "chunks_created": chunks}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.get("/languages")
def list_languages():
    """Return all supported response languages for multilingual Q&A."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    # Validate language code
    lang_code = req.language.lower().strip() if req.language else "en"
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, "English")

    try:
        rag_chain = _get_rag_chain_module()
        result = rag_chain.ask_question(
            req.question,
            GLOBAL_COLLECTION,
            top_k=8,
            response_language=lang_name,
        )
        elapsed = int((time.time() - start) * 1000)
        source_items = [SourceItem(**s) for s in result.get("sources", [])]
        return AskResponse(answer=result.get("answer", ""), sources=source_items, processing_time_ms=elapsed)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ask/stream")
def ask_stream(question: str):
    """Server-Sent Events streaming endpoint — returns tokens as they generate."""
    def event_generator():
        try:
            rag_chain = _get_rag_chain_module()
            for token in rag_chain.stream_answer(question, GLOBAL_COLLECTION):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse)
def health():
    qdrant_ok = False
    try:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if url and api_key:
            rag_chain = _get_rag_chain_module()
            client = rag_chain._get_qdrant_client()
            _ = client.get_collections()
            qdrant_ok = True
    except Exception as exc:
        logger.exception("Qdrant health check failed: %s", exc)
        qdrant_ok = False

    return {"status": "ok", "qdrant_connected": qdrant_ok}


@app.post("/revision", response_model=RevisionResponse)
def revision(req: TopicRequest):
    from langchain_core.messages import SystemMessage, HumanMessage
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.topic, top_k=8)

    system = (
        "You are a college exam revision assistant. "
        "Write clearly structured revision sheets using markdown-style formatting: "
        "use ## for section headers, **bold** for key terms, and bullet points (- item) for lists."
    )
    prompt = (
        f"Topic: {req.topic}\n\n"
        f"Context from notes:\n{joined_context}\n\n"
        "Write a revision sheet with EXACTLY these sections:\n"
        "## Definition\n"
        "## Key Points\n"
        "(5 bullet points, each starting with -)\n"
        "## Formulas / Rules\n"
        "(list each formula on its own line)\n"
        "## Common Exam Traps\n"
        "(2-3 common mistakes students make)\n\n"
        "Cite page numbers inline like [Page 4]. Use only the provided notes."
    )
    llm = _get_llm()
    answer = (llm | StrOutputParser()).invoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    elapsed = int((time.time() - start) * 1000)
    return {
        "revision_sheet": answer,
        "sources": [SourceItem(**s) for s in sources],
        "processing_time_ms": elapsed,
    }


@app.post("/quiz", response_model=QuizResponse)
def quiz(req: QuizRequest):
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.topic, top_k=8)

    system = "You generate strict JSON only. No markdown. No commentary."
    prompt = (
        f"Topic: {req.topic}\n\n"
        f"Context:\n{joined_context}\n\n"
        f"Generate {req.num_questions} multiple-choice questions.\n"
        "Rules:\n"
        "- Each question must be answerable from the notes.\n"
        "- Each question must have exactly 4 options.\n"
        "- Only one option is correct.\n"
        "- Include a short explanation.\n"
        "Return ONLY valid JSON with shape {{\"quiz\": [{{\"question\": \"string\", \"options\": [\"string\", \"string\", \"string\", \"string\"], \"correct\": \"string\", \"explanation\": \"string\", \"source_page\": 1}}]}}"
    )

    parsed = None
    last_error = None
    for _ in range(3):
        try:
            parsed = _invoke_json_llm(prompt=prompt, system=system, retries=0)
            if not isinstance(parsed, dict) or "quiz" not in parsed:
                raise ValueError("Missing quiz key")
            quiz_items = parsed["quiz"]
            if not isinstance(quiz_items, list):
                raise ValueError("quiz must be a list")
            validated = []
            for item in quiz_items:
                validated.append(QuizItem(**item))
            elapsed = int((time.time() - start) * 1000)
            return {"quiz": validated, "processing_time_ms": elapsed}
        except Exception as exc:
            last_error = exc
            logger.warning("Quiz JSON validation failed, retrying: %s", exc)
    raise HTTPException(status_code=500, detail=f"Could not generate valid quiz JSON: {last_error}")


@app.post("/explain-simple", response_model=SimpleExplanationResponse)
def explain_simple(req: ConceptRequest):
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.concept, top_k=8)

    system = "You explain concepts to a new college student using plain English."
    prompt = (
        f"Concept: {req.concept}\n\n"
        f"Context:\n{joined_context}\n\n"
        "Return ONLY valid JSON matching this exact shape: {\"simple_explanation\": \"string\", \"analogy\": \"string\", \"one_thing_to_remember\": \"string\"}.\n"
        "Ensure all string values are fully populated with your detailed explanation.\n"
        "Avoid jargon. If you must use a technical term, define it immediately."
    )

    try:
        parsed = _invoke_json_llm(prompt=prompt, system=system, retries=2)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        elapsed = int((time.time() - start) * 1000)
        return {
            "simple_explanation": str(parsed.get("simple_explanation", "")),
            "analogy": str(parsed.get("analogy", "")),
            "one_thing_to_remember": str(parsed.get("one_thing_to_remember", "")),
            "sources": [SourceItem(**s) for s in sources],
            "processing_time_ms": elapsed,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/audio-notes")
def audio_notes(req: TopicRequest):
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.topic, top_k=8)

    try:
        from gtts import gTTS
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"gTTS is not available: {exc}")

    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _get_llm()
    system = (
        "You write spoken audio revision scripts. "
        "Output ONLY the spoken words — never include stage directions, tone instructions, "
        "timestamps, parenthetical notes like (0:04), (gentle tone), or any meta-commentary. "
        "The output will be read aloud directly by a text-to-speech engine."
    )
    prompt = (
        f"Write a clear, engaging spoken revision script of about 250 words on: {req.topic}\n\n"
        f"Use only these notes as source material:\n{joined_context}\n\n"
        "Start immediately with the topic content. No introductory filler."
    )
    script = (llm | StrOutputParser()).invoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    if not hasattr(script, '__len__'):
        script = str(script)

    if len(script.strip()) < 10:
        script = f"I am sorry, but I was unable to generate a revision script for {req.topic} based on the notes."

    audio_buffer = Path("/tmp") / f"{uuid.uuid4()}.mp3"
    tts = gTTS(text=script, lang="en", slow=False)
    tts.save(str(audio_buffer))
    audio_bytes = audio_buffer.read_bytes()
    try:
        audio_buffer.unlink(missing_ok=True)
    except Exception:
        pass

    elapsed = int((time.time() - start) * 1000)
    headers = {
        "Content-Disposition": f'attachment; filename="{req.topic.replace(" ", "-")}.mp3"',
        "X-Processing-Time-MS": str(elapsed),
    }
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)


@app.post("/night-before", response_model=NightBeforeResponse)
def night_before(req: NightBeforeRequest):
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.subject, top_k=8)

    system = "You create ruthless exam cram sheets."
    prompt = (
        f"Subject: {req.subject}\nExam in {req.exam_hours_away} hours\n\n"
        f"Context:\n{joined_context}\n\n"
        "Return ONLY valid JSON matching this exact shape: {\"cheat_sheet\": \"string\", \"topics_covered\": [\"string\", \"string\"]}.\n"
        "Prioritize the top 5 topics, must-remember facts, and common confusions."
    )
    try:
        parsed = _invoke_json_llm(prompt=prompt, system=system, retries=2)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        elapsed = int((time.time() - start) * 1000)
        return {
            "cheat_sheet": str(parsed.get("cheat_sheet", "")),
            "topics_covered": _ensure_list_of_strings(parsed.get("topics_covered", [])),
            "sources": [SourceItem(**s) for s in sources],
            "processing_time_ms": elapsed,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/flashcards", response_model=FlashcardsResponse)
def flashcards(req: FlashcardsRequest):
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.topic, top_k=8)

    system = "You generate flashcards as strict JSON only."
    prompt = (
        f"Topic: {req.topic}\n\n"
        f"Context:\n{joined_context}\n\n"
        f"Generate {req.num_cards} flashcards. Return JSON with key flashcards.\n"
        "Each flashcard must have front, back, difficulty, source_page.\n"
        "difficulty must be easy, medium, or hard."
    )
    try:
        parsed = _invoke_json_llm(prompt=prompt, system=system, retries=2)
        if not isinstance(parsed, dict) or "flashcards" not in parsed:
            raise ValueError("Missing flashcards key")
        flashcards_data = [FlashcardItem(**item) for item in parsed["flashcards"]]
        elapsed = int((time.time() - start) * 1000)
        return {"flashcards": flashcards_data, "processing_time_ms": elapsed}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/highlights", response_model=HighlightsResponse)
def highlights(req: HighlightsRequest):
    start = time.time()
    _collection_exists()
    sources, joined_context = _retrieve_sources(req.topic, top_k=8)

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", joined_context) if part.strip()]
    sentences = sentences[:40]

    system = "You score sentences for exam importance and return strict JSON only."
    prompt = (
        f"Topic: {req.topic}\n\n"
        f"Sentences:\n" + "\n".join(f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences)) + "\n\n"
        "Return ONLY valid JSON matching this exact shape: {\"highlights\": [{\"sentence\": \"string\", \"score\": 10, \"reason\": \"string\", \"page_number\": 1}]}."
    )
    try:
        parsed = _invoke_json_llm(prompt=prompt, system=system, retries=2)
        if not isinstance(parsed, dict) or "highlights" not in parsed:
            raise ValueError("Missing highlights key")
        highlights_data = [HighlightItem(**item) for item in parsed["highlights"]]
        highlights_data = sorted(highlights_data, key=lambda item: item.score, reverse=True)[:10]
        elapsed = int((time.time() - start) * 1000)
        return {"highlights": highlights_data, "processing_time_ms": elapsed}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Frontend Static Serving ---
# This must be at the end of the file so API routes take priority.
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.isdir(frontend_dist):
    # Mount the assets directory specifically
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    
    # Catch-all for SPA routing (e.g. React Router)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(path):
            return FileResponse(path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    logger.warning("Frontend dist directory not found. Static React files will not be served.")
