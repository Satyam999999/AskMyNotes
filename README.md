<div align="center">

# AskMyNotes 🧠

### Production-Grade RAG System for Academic Documents

*Upload any PDF. Ask anything. Get cited answers in under 3 seconds.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC143C?style=for-the-badge)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)](https://groq.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Traced-FF6B35?style=for-the-badge)](https://smith.langchain.com)

</div>

---

## What This Is

Students carry hundreds of pages of study material — PDFs, lecture slides, textbook chapters — with no way to query them intelligently.
AskMyNotes is a **production-grade RAG system** that lets you upload your own PDFs and ask questions in plain English, returning answers grounded in your documents with exact page citations.

This is not a wrapper around an LLM API. Every component — retrieval, ranking, evaluation, observability — is engineered and measured.

Now with **Multilingual Q&A** — ask questions and receive answers in English or Japanese (日本語).

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                    INGESTION PIPELINE                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  PDF Upload                                                          ║
║      │                                                               ║
║      ▼                                                               ║
║  pymupdf4llm ──► Page-by-Page Markdown   ← preserves tables,        ║
║  (page_chunks=True)                        headings, reading order   ║
║      │                                                               ║
║      ▼                                                               ║
║  MarkdownHeaderTextSplitter                                          ║
║  + RecursiveCharacterTextSplitter         chunk_size=800, overlap=100║
║      │                                                               ║
║      ├──► Dense Embedding  (BAAI/bge-small-en-v1.5 via FastEmbed)   ║
║      └──► Sparse Embedding (SPLADE prithivida/Splade_PP_en_v1)      ║
║               │                                                      ║
║               ▼                                                      ║
║         Qdrant Cloud  ──  Named Vectors: {dense, sparse}            ║
║                           Payload: {text, page_number, source_file} ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                    QUERY PIPELINE                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  User Question (any language)                                        ║
║      │                                                               ║
║      ├──► Dense Query Vector  ─────────────────────────┐            ║
║      └──► Sparse Query Vector  ────────────────────────┤            ║
║                                                         ▼            ║
║                                            Qdrant Hybrid Search      ║
║                                            RRF Fusion (top-25)       ║
║                                                         │            ║
║                                                         ▼            ║
║                                            FlashRank Cross-Encoder   ║
║                                            ms-marco-MiniLM-L-12-v2  ║
║                                            (local ONNX, zero cost)   ║
║                                                         │            ║
║                                            Re-ranked top-5 chunks    ║
║                                                         │            ║
║                                                         ▼            ║
║                                            Groq LLM (multilingual)   ║
║                                            Citation system prompt    ║
║                                                         │            ║
║                            LangSmith ◄─────────────────┤  traced    ║
║                                                         ▼            ║
║                                     Answer + Page Citations          ║
║                                     (in your chosen language)        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## What We Built & Measured

This system went through two engineering iterations. Every change was measured using **RAGAS** (Retrieval-Augmented Generation Assessment) on 8 golden Q&A pairs.

### RAGAS Results — v1 Baseline → v2 Production

| Metric | v1 Baseline | v2 Production | Improvement |
|--------|:-----------:|:-------------:|:-----------:|
| **Faithfulness** *(hallucination resistance)* | 0.425 | **0.934** | **+119.7%** |
| **Answer Relevancy** *(response precision)* | 0.504 | **0.897** | **+78.1%** |
| **Context Recall** *(retrieval completeness)* | 0.465 | **0.927** | **+99.5%** |

### Latency Optimization ⚡

| Endpoint | Before | After | Speedup |
|----------|:------:|:-----:|:-------:|
| **`/ask` (Full RAG)** | 19.0s | **2.9s** | **84% Faster** |
| **`/flashcards`** | 4.5s | **2.5s** | **44% Faster** |
| **`/night-before`** | 4.5s | **2.9s** | **35% Faster** |
| **`/quiz`** | 2.7s | **1.7s** | **37% Faster** |

---

### Engineering Changes That Drove the Improvement

#### 1. Semantic PDF Parsing → Faithfulness +119%
**Problem:** The v1 pipeline used `PyMuPDFLoader` which reads PDFs as a raw character stream. Two-column lecture slides and tables came out as scrambled, out-of-order text.

**Solution:** Replaced with `pymupdf4llm.to_markdown(pdf, page_chunks=True)`. This converts each page to structured Markdown, preserving headers (`##`), tables, and reading order — and returns the exact `page_number` for every chunk.

```python
# Before — raw stream, no structure
pages = PyMuPDFLoader(path).load()

# After — page-aware Markdown with layout preservation
pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
# Each page: {"text": "## Supervised Learning\n...", "metadata": {"page_number": 3}}
```

---

#### 2. Embedding Model Upgrade → Better Semantic Clustering
**Problem:** `all-MiniLM-L6-v2` underperforms on technical/academic text.

**Solution:** Switched to `BAAI/bge-small-en-v1.5`, top-ranked on BEIR retrieval benchmarks, deployed via FastEmbed (ONNX runtime, no GPU required).

---

#### 3. Hybrid Search: Dense + Sparse via RRF → Context Recall +99%
**Problem:** Dense-only vector search misses exact keyword queries.

**Solution:** Store both dense (BGE) and sparse (SPLADE) vectors per chunk in Qdrant's named vector spaces. At query time, run both searches and fuse via **Reciprocal Rank Fusion (RRF)**.

```python
results = client.query_points(
    collection_name=collection,
    prefetch=[
        Prefetch(query=dense_vec,  using="dense",  limit=25),
        Prefetch(query=sparse_vec, using="sparse", limit=25),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=25,
)
```

---

#### 4. Cross-Encoder Re-Ranking → Answer Relevancy +78%
**Problem:** Cosine similarity ranks chunks by embedding proximity, not by how directly they answer the question.

**Solution:** Fetch top-25 candidates from Qdrant, pass to **FlashRank** (`ms-marco-MiniLM-L-12-v2`). The cross-encoder directly scores each `(question, chunk)` pair. Only the top-5 re-ranked chunks go to the LLM.

```
Qdrant (25 candidates) → FlashRank cross-encoder → top-5 → LLM
```

Zero API cost. Runs locally in ~50ms on CPU.

---

#### 5. Chunk Size Optimisation
**Problem:** 500-character chunks frequently split mid-sentence, separating definitions from their explanations.

**Solution:** Increased to 800 characters with 100-character overlap, combined with header-aware splitting.

---

## LangSmith Observability

Every LLM call is automatically traced via LangSmith — latency, token counts, retrieved chunks, and full prompt are visible per request.

---

## Multilingual Q&A 🌐

Ask questions and receive answers in your preferred language. The system retrieves context from your documents (always in the original PDF language) and generates responses in your chosen language.

| Code | Language |
|------|----------|
| `en` | 🇬🇧 English |
| `ja` | 🇯🇵 日本語 (Japanese) |

Example — Japanese response:
```json
POST /ask
{ "question": "What is this document about?", "language": "ja" }
```

---

## 8 Study Modes — One RAG Pipeline

All modes share the same Qdrant index and retrieval pipeline. Different prompts produce different structured outputs.

| Mode | Endpoint | What It Does |
|------|----------|-------------|
| 💬 **Ask a Question** | `/ask` | Cited answer with exact page references, multilingual |
| 📋 **Revision Sheet** | `/revision` | Definition + key points + formulas + exam traps |
| 🧠 **Auto Quiz** | `/quiz` | MCQs with options, correct answer, explanation |
| 🗣️ **Explain Simply** | `/explain-simple` | Complex concept rephrased with analogies |
| 🎧 **Audio Notes** | `/audio-notes` | Spoken MP3 revision notes via gTTS |
| 🌙 **Night Before Exam** | `/night-before` | Ultra-condensed cram sheet for a subject |
| 🃏 **Flashcards** | `/flashcards` | Anki-style front/back pairs with difficulty tags |
| ✨ **Smart Highlights** | `/highlights` | Top scoring sentences for revision |

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **LLM** | Groq API | Fast inference, multilingual support |
| **Dense Embeddings** | `BAAI/bge-small-en-v1.5` via FastEmbed | BEIR benchmark leader for passage retrieval |
| **Sparse Embeddings** | SPLADE `prithivida/Splade_PP_en_v1` | Exact keyword matching complement to dense |
| **Re-Ranker** | FlashRank `ms-marco-MiniLM-L-12-v2` | Local ONNX cross-encoder, zero latency overhead |
| **Vector DB** | Qdrant Cloud | Native named-vector hybrid search + RRF |
| **PDF Parser** | `pymupdf4llm` | Page-aware Markdown preserving layout |
| **Backend** | FastAPI + Uvicorn | Async, typed, production-grade Python API |
| **Frontend** | React + Vite | NotebookLM-inspired 8-mode study interface |
| **Evaluation** | RAGAS | Faithfulness, answer relevancy, context recall |
| **Observability** | LangSmith | Full chain tracing on every request |

---

## Setup

### Prerequisites
- Python 3.11+, Node 18+
- [Qdrant Cloud](https://cloud.qdrant.io) cluster
- [Groq](https://console.groq.com) API key
- [LangSmith](https://smith.langchain.com) API key (optional, for tracing)

### Run Locally

```bash
git clone https://github.com/Satyam999999/AskMyNotes.git
cd AskMyNotes
cp .env.example .env   # fill in your keys
```

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### `.env` reference

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_preferred_groq_model
LANGCHAIN_API_KEY=your_langsmith_key       # optional
LANGCHAIN_TRACING_V2=true                  # optional
```

---

## API Reference

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/health` | GET | — | `{qdrant_connected}` |
| `/languages` | GET | — | `{languages: {code: name}}` |
| `/upload` | POST | PDF file | `{collection_id, chunks_created}` |
| `/ask` | POST | `{question, language?}` | `{answer, sources[], processing_time_ms}` |
| `/revision` | POST | `{topic}` | `{revision_sheet, sources[]}` |
| `/quiz` | POST | `{topic, num_questions}` | `{quiz[]}` |
| `/explain-simple` | POST | `{concept}` | `{simple_explanation, analogy, one_thing_to_remember}` |
| `/audio-notes` | POST | `{topic}` | MP3 binary |
| `/night-before` | POST | `{subject, exam_hours_away}` | `{cheat_sheet, topics_covered}` |
| `/flashcards` | POST | `{topic, num_cards}` | `{flashcards[]}` |
| `/highlights` | POST | `{topic}` | `{highlights[]}` |

---

## Project Structure

```
AskMyNotes/
├── app/
│   ├── main.py          # FastAPI — 10 endpoints, multilingual support
│   ├── rag_chain.py     # Hybrid search + FlashRank re-ranking + Groq LLM
│   └── ingest.py        # PDF parsing + chunking + dual embedding + Qdrant upload
├── frontend/
│   └── src/
│       ├── App.jsx      # React — 8 study modes, language toggle, upload progress
│       └── App.css      # Design system — glassmorphism, dark mode
├── eval/
│   ├── eval_set.json         # 8 golden Q&A pairs
│   ├── evaluate_v2.py        # RAGAS pipeline runner
│   ├── results_baseline.csv  # v1 scores
│   ├── results_improved.csv  # v2 scores
│   ├── dashboard.py          # Interactive Streamlit dashboard & ablation charts
│   └── improvement_report.md
├── docs/
│   └── RAG_EVOLUTION.md      # Full engineering journey & architectural decisions
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## License

MIT © 2026 Satyam Ghosh
