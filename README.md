<div align="center">

# AskMyNotes 🧠

### Production-Grade RAG System for Academic Documents

*Upload any PDF. Ask anything. Get cited answers in under 2 seconds.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC143C?style=for-the-badge)](https://qdrant.tech)
[![GCP](https://img.shields.io/badge/GCP_Cloud_Run-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![LangSmith](https://img.shields.io/badge/LangSmith-Traced-FF6B35?style=for-the-badge)](https://smith.langchain.com)

</div>

---

## What This Is

Students carry hundreds of pages of study material — PDFs, lecture slides, textbook chapters — with no way to query them intelligently.
AskMyNotes is a **production-deployed RAG system** that lets you upload your own PDFs and ask questions in plain English, returning answers grounded in your documents with exact page citations.

This is not a wrapper around an LLM API. Every component — retrieval, ranking, evaluation, observability, deployment — is engineered and measured.

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
║  User Question                                                       ║
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
║                                            Groq LLM (llama-3.3-70b) ║
║                                            Citation system prompt    ║
║                                                         │            ║
║                            LangSmith ◄─────────────────┤  traced    ║
║                                                         ▼            ║
║                                            Answer + Page Citations   ║
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

### Latency Optimization (Before vs. After) ⚡
We aggressively profiled and optimized the pipeline to deliver interactive, sub-3-second latencies on a Serverless environment by tuning the Cross-Encoder context window and LLM parameters.

| Endpoint | Before (Unoptimized) | After (Optimized) | Speedup |
|----------|:--------------------:|:-----------------:|:-------:|
| **`/ask` (Full RAG)** | 19.0s | **2.9s** | **84% Faster** |
| **`/flashcards`** | 4.5s | **2.5s** | **44% Faster** |
| **`/night-before`** | 4.5s | **2.9s** | **35% Faster** |
| **`/quiz`** | 2.7s | **1.7s** | **37% Faster** |

### RAGAS Metrics Dashboard & Ablation Study 📊

To systematically analyse the improvements, we built a premium, interactive Streamlit evaluation dashboard. It visualises before/after KPI deltas, a capability radar chart, incremental ablation paths showing the cumulative impact of each engineering choice, a question-by-question faithfulness heatmap, and an end-to-end latency breakdown by study mode.

![RAGAS Evaluation Dashboard](docs/metrics_dashboard.png)

#### Running the Dashboard locally
To launch and explore the interactive metrics and ablation charts:
```bash
# Install visualization dependencies
pip install streamlit plotly pandas

# Run the Streamlit server
streamlit run eval/dashboard.py
```

---

### Engineering Changes That Drove the Improvement

#### 1. Semantic PDF Parsing → Faithfulness +119%
**Problem:** The v1 pipeline used `PyMuPDFLoader` which reads PDFs as a raw character stream. Two-column lecture slides and tables came out as scrambled, out-of-order text. The LLM was building answers from garbled context — causing hallucinations.

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
**Problem:** `all-MiniLM-L6-v2` is a general-purpose model that underperforms on technical/academic text — it struggles to differentiate between similar ML concepts.

**Solution:** Switched to `BAAI/bge-small-en-v1.5`, which is top-ranked on BEIR retrieval benchmarks and specifically optimised for dense passage retrieval. Deployed via FastEmbed (ONNX runtime, no GPU required).

---

#### 3. Hybrid Search: Dense + Sparse via RRF → Context Recall +99%
**Problem:** Dense-only vector search misses exact keyword queries. When a student asks *"What is the formula for F1 score?"*, the dense vector for "F1 score" in a technical document is poorly defined — the retriever returns semantically similar but wrong chunks.

**Solution:** Store both dense (BGE) and sparse (SPLADE) vectors per chunk in Qdrant's named vector spaces. At query time, run both searches and fuse via **Reciprocal Rank Fusion (RRF)** — capturing conceptual similarity AND exact keyword matches simultaneously.

```python
# Qdrant Hybrid Query with RRF fusion
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
**Problem:** Vector similarity scores are a blunt instrument. Cosine similarity ranks chunks by embedding proximity, not by how directly they answer the question. The top-8 results frequently included off-topic sections that diluted LLM context.

**Solution:** Fetch top-25 candidates from Qdrant, then pass all 25 to **FlashRank** (local ONNX cross-encoder `ms-marco-MiniLM-L-12-v2`). The cross-encoder directly scores each `(question, chunk)` pair — far more precise than cosine similarity. Only the top-5 re-ranked chunks go to the LLM.

```
Qdrant (25 candidates) → FlashRank cross-encoder → top-5 → LLM
```

Zero API cost. Runs locally in ~50ms on CPU.

---

#### 5. Chunk Size Optimisation
**Problem:** 500-character chunks frequently split mid-sentence, separating definitions from their explanations and formulas from their context.

**Solution:** Increased to 800 characters with 100-character overlap, combined with header-aware splitting. Chunks stay within logical section boundaries.

---

## LangSmith Observability

Every LLM call is automatically traced via LangSmith — latency, token counts, retrieved chunks, and full prompt are visible per request. All 8 study mode chains show sub-1.5s end-to-end latency.

![LangSmith Traces](docs/langsmith_traces.png)

---

## 7 Study Modes — One RAG Pipeline

All modes share the same Qdrant index and retrieval pipeline. Different prompts produce different structured outputs.

| Mode | What It Does |
|------|-------------|
| 💬 **Ask a Question** | Cited answer with exact page references |
| 📋 **Revision Sheet** | Definition + 5 key points + formulas + exam traps |
| 🧠 **Auto Quiz** | MCQs with options, correct answer, explanation |
| 🗣️ **Explain Simply** | Complex concept rephrased with analogies |
| 🎧 **Audio Notes** | Spoken MP3 revision notes via gTTS |
| 🌙 **Night Before Exam** | Ultra-condensed cram sheet for a subject |
| 🃏 **Flashcards** | Anki-style front/back pairs with difficulty tags |

---

## Sample Q&A

**Upload:** `supervised-learning-lecture.pdf`

**Question:** *"What is the difference between supervised and unsupervised learning?"*

**Answer:**
```
Supervised learning trains a model on labeled examples — each input has a
known correct output — allowing it to learn a mapping from inputs to labels
(e.g. predicting house prices from square footage).

Unsupervised learning discovers hidden structure in data without labels,
such as grouping customers into segments based on purchase patterns.

Sources:
  • Page 3 — supervised-learning-lecture.pdf  (definition + training set diagram)
  • Page 7 — supervised-learning-lecture.pdf  (comparison table)
```

**Latency:** 1.24s end-to-end

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **LLM** | Groq `llama-3.3-70b-versatile` | Fastest inference API, sub-second tokens |
| **Dense Embeddings** | `BAAI/bge-small-en-v1.5` via FastEmbed | BEIR benchmark leader for passage retrieval |
| **Sparse Embeddings** | SPLADE `prithivida/Splade_PP_en_v1` | Exact keyword matching complement to dense |
| **Re-Ranker** | FlashRank `ms-marco-MiniLM-L-12-v2` | Local ONNX cross-encoder, zero latency overhead |
| **Vector DB** | Qdrant Cloud | Native named-vector hybrid search + RRF |
| **PDF Parser** | `pymupdf4llm` | Page-aware Markdown preserving layout |
| **Backend** | FastAPI + Uvicorn | Async, typed, production-grade Python API |
| **Frontend** | React + Vite | NotebookLM-inspired 7-mode study interface |
| **Evaluation** | RAGAS | Faithfulness, answer relevancy, context recall |
| **Observability** | LangSmith | Full chain tracing on every request |
| **Deployment** | GCP Cloud Run | Containerised, auto-scaling, CI/CD via GitHub Actions |

---

## Deployment — GCP Cloud Run

Backend deployed on GCP Cloud Run with auto-scaling and GitHub Actions CI/CD.

### The Serverless ML DevOps Saga
Deploying heavy AI pipelines (Dense + Sparse embeddings, Cross-Encoders, PyMuPDF) to Serverless infrastructure required exhaustive DevOps debugging:
- **Unified Container (CORS fix):** Wrote a Multi-Stage Dockerfile to compile React/Vite into static assets and mount them natively inside FastAPI, completely eliminating cross-origin issues.
- **Bypassing Ephemeral Storage Timeouts:** `fastembed` failed to download gigabytes of AI models at runtime into Cloud Run's temporary RAM drive. We solved this by executing a Python caching script *during the Docker build step*, baking the models directly into `/opt/` inside the image.
- **Solving OOM (Out-of-Memory) Crashes:** When the models successfully loaded instantly, loading them alongside `PyMuPDF` caused a massive RAM spike that exceeded the 2GB container limit (503 Service Unavailable). We tuned the Cloud Run YAML to provision 4GB RAM + 2 vCPUs to perfectly accommodate the local ML stack.
- **Qdrant Schema Migrations:** Upgrading from Dense-only to Hybrid Search caused Qdrant to reject payloads. We bumped the `GLOBAL_COLLECTION` version to automatically spin up a fresh, perfectly configured database schema with zero downtime.

Read the full, in-depth architectural journey in [docs/RAG_EVOLUTION.md](docs/RAG_EVOLUTION.md) and [docs/GCP_DEPLOYMENT_SAGA.md](docs/GCP_DEPLOYMENT_SAGA.md).

```bash
# Dockerfile — CPU-only PyTorch to avoid CUDA bloat
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Cloud Run reads $PORT at runtime — shell form required
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
```

**CI/CD pipeline** (`.github/workflows/deploy.yml`):
- Authenticates via **Workload Identity Federation** — no service account key files stored
- Builds Docker image with GitHub Actions cache (layer caching)
- Pushes to Artifact Registry
- Deploys to Cloud Run on every push to `main`

---

## Setup

### Prerequisites
- Python 3.11+, Docker, Node 18+
- [Qdrant Cloud](https://cloud.qdrant.io) cluster
- [Groq](https://console.groq.com) API key
- [LangSmith](https://smith.langchain.com) API key

### Run locally

```bash
git clone https://github.com/Satyam999999/AskMyNotes.git
cd AskMyNotes
cp .env.example .env   # fill in your keys
```

```bash
# Backend
docker build -t askmynotes .
docker run -p 8080:8080 --env-file .env askmynotes

# Frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### `.env` reference

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

---

## API Reference

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/health` | GET | — | `{qdrant_connected, model_loaded}` |
| `/upload` | POST | PDF file | `{collection_id, chunks_created}` |
| `/ask` | POST | `{question}` | `{answer, sources[], processing_time_ms}` |
| `/revision` | POST | `{topic}` | `{revision_sheet}` |
| `/quiz` | POST | `{topic, num_questions}` | `{quiz[]}` |
| `/explain-simple` | POST | `{concept}` | `{simple_explanation, analogy, one_thing_to_remember}` |
| `/audio-notes` | POST | `{topic}` | MP3 binary |
| `/night-before` | POST | `{subject, exam_hours_away}` | `{cheat_sheet, topics_covered}` |
| `/flashcards` | POST | `{topic, num_cards}` | `{flashcards[]}` |
| `/highlights` | POST | `{topic}` | `{highlights[{sentence, reason, page_number}]}` |

---

## Project Structure

```
AskMyNotes/
├── app/
│   ├── main.py          # FastAPI — 10 endpoints, SQLite metadata store
│   ├── rag_chain.py     # Hybrid search + FlashRank re-ranking + Groq LLM
│   └── ingest.py        # PDF parsing + chunking + dual embedding + Qdrant upload
├── frontend/
│   └── src/
│       ├── App.jsx      # React — 7 study modes, upload progress, processing states
│       └── App.css      # Design system — glassmorphism, dark mode
├── eval/
│   ├── eval_set.json         # 8 golden Q&A pairs
│   ├── evaluate_v2.py        # RAGAS pipeline runner
│   ├── results_baseline.csv  # v1 scores
│   ├── results_improved.csv  # v2 scores
│   ├── dashboard.py          # Interactive Streamlit dashboard & ablation charts
│   └── improvement_report.md
├── docs/
│   ├── langsmith_traces.png  # LangSmith execution traces
│   └── metrics_dashboard.png # RAGAS metrics & ablation dashboard
├── Dockerfile
├── .dockerignore
├── .github/workflows/deploy.yml
└── requirements.txt
```

---

## License

MIT © 2026 Satyam Ghosh
