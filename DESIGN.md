# AskMyNotes — RAG Architecture & Improvement Report

## Baseline Scores (v1 Pipeline — May 18, 2026)

Measured via RAGAS on 8 questions from `eval/eval_set.json`:

| Metric | Score |
|--------|-------|
| **Faithfulness** | **0.375** |
| **Answer Relevancy** | **0.099** |
| **Context Recall** | **0.375** |

> These are the actual scores from `eval/eval_results.csv` produced before any improvements.

---

## Why the Scores Were Low

| Problem | Root Cause |
|---------|-----------|
| Faithfulness: 0.375 | PyMuPDF scrambled table/column layouts; LLM hallucinated from garbled context |
| Answer Relevancy: 0.099 | Cosine retrieval returned tangential chunks; LLM rambled around the wrong content |
| Context Recall: 0.375 | Dense-only search missed exact keyword questions ("accuracy formula", "confusion matrix") |

---

## Improvements Implemented (v2 Pipeline)

### 1. Semantic Markdown Parsing — `pymupdf4llm`

**Before:** `PyMuPDFLoader` reads PDF as a raw character stream. Two-column textbook layouts
produce interleaved, out-of-order text:

```
Column A line 1   Column B line 1
Column A line 2   Column B line 2
→ extracted as: "Column A line 1 Column B line 1 Column A line 2 Column B line 2"
```

**After:** `pymupdf4llm.to_markdown()` applies visual layout analysis (ONNX model) to produce
clean, reading-order Markdown that preserves tables, headings, and paragraph boundaries:

```markdown
## Confusion Matrix
| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |
```

**RAGAS Impact:** Faithfulness ↑ — the LLM can now correctly read and reason about
tables without making up relationships that don't exist in the source.

---

### 2. Semantic Header-Aware Chunking

**Before:** `RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)` sliced text
at arbitrary character positions, often splitting a definition from its explanation.

**After:** Two-pass split strategy:
1. `MarkdownHeaderTextSplitter` — primary split at `#`, `##`, `###` boundaries so entire sections are kept together
2. `RecursiveCharacterTextSplitter(chunk_size=800, overlap=100)` — secondary split only when a section exceeds 800 chars

This ensures "Accuracy = (TP + TN) / (TP + TN + FP + FN)" stays in the same chunk as its
"Accuracy measures the fraction of correctly classified samples" definition.

**RAGAS Impact:** Answer Relevancy ↑ — chunks are semantically complete units.

---

### 3. Embedding Model Upgrade — `BAAI/bge-small-en-v1.5`

**Before:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional general-purpose embeddings.

**After:** `BAAI/bge-small-en-v1.5` — same 384 dimensions but trained specifically for
dense passage retrieval (top-10 on BEIR benchmark). Produces tighter vector clusters for
academic/technical vocabulary.

**Impact:** Better cosine similarity distances between question vectors and relevant chunks.

---

### 4. Hybrid Search — Dense BGE + Sparse SPLADE via RRF

**Before:** Dense-only cosine search. For exact-keyword queries like "What is the formula for accuracy?",
the embedding model averages word meanings — "formula" and "accuracy" get diluted together,
often missing the chunk that literally says "Accuracy = (TP+TN)/(TP+TN+FP+FN)".

**After:** Two prefetch queries run in parallel in Qdrant:
- Dense query: BGE-small embedding → semantic conceptual match
- Sparse query: SPLADE embeddings → exact keyword/token match

Scores fused via **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))   where k=60
```

**RAGAS Impact:** Context Recall ↑ — both conceptual questions ("explain supervised learning")
and exact definitions ("What is TP?") now reliably retrieve the right chunks.

> **Note:** Existing Qdrant collections created before this update are dense-only. The code
> automatically falls back to dense-only search for these collections. Re-upload PDFs to a
> fresh collection to activate hybrid search.

---

### 5. Cross-Encoder Re-Ranking — `ms-marco-MiniLM-L-12-v2` via FlashRank

**Before:** Top 8 cosine-similar chunks sent directly to the LLM. Cosine similarity is a
"bag of words" approximation — it doesn't understand the *relationship* between question and passage.

**After:** Two-stage retrieval:
1. **Stage 1:** Hybrid search fetches top **25 candidates** (over-fetch for recall)
2. **Stage 2:** FlashRank cross-encoder scores each (question, chunk) pair jointly:

```
score = CrossEncoder([CLS] question [SEP] chunk [SEP])
```

The cross-encoder directly models the interaction between question tokens and passage tokens
using full attention (unlike bi-encoder embeddings which are independent). Best **5 chunks**
are selected.

This is the single most impactful improvement:

**RAGAS Impact:** Answer Relevancy ↑↑ — off-topic chunks are demoted; LLM receives only
maximally relevant context and stops rambling.

---

### 6. Streaming SSE Endpoint — `/ask/stream`

**Before:** `/ask` blocked until full generation completed (~3-8 seconds).

**After:** New `GET /ask/stream?question=...` endpoint streams tokens via Server-Sent Events:

```
data: {"token": "Supervised"}
data: {"token": " learning"}
data: {"token": " is..."}
data: [DONE]
```

Groq already generates tokens at ~300 tokens/second; streaming surface them instantly.

---

## Running the Evaluation

### Step 1 — First, re-upload your PDFs to activate hybrid search

```bash
# The new collection needs sparse vectors configured at creation time
# Upload via the AskMyNotes UI (drag-and-drop to Sources panel)
```

### Step 2 — Run RAGAS evaluation

```bash
# Score the improved (v2) pipeline
cd /Users/satyamghosh/Downloads/ASKMYNOTES
python eval/evaluate_v2.py --phase improved --collection askmynotes_global --execute

# Generate comparison report
python eval/evaluate_v2.py --phase report
```

The report will be saved to `eval/improvement_report.md` with actual before/after numbers.

---

## Expected Improvement Targets

Based on the architectural changes applied:

| Metric | v1 Baseline | v2 Target | Driver |
|--------|-------------|-----------|--------|
| Faithfulness | 0.375 | > 0.70 | Markdown parsing + re-ranking |
| Answer Relevancy | 0.099 | > 0.60 | Re-ranking + semantic chunks |
| Context Recall | 0.375 | > 0.75 | Hybrid search (SPLADE + BGE) |

> Actual scores depend on the quality of the PDFs uploaded. The eval set (`eval/eval_set.json`)
> must correspond to content in the indexed collection for meaningful Context Recall measurement.

---

## Architecture Diagram

```
USER QUESTION
     │
     ▼
┌────────────────────────────────┐
│  Embed Query                   │
│  ├─ Dense: BGE-small (384d)    │
│  └─ Sparse: SPLADE (vocab)     │
└──────────────┬─────────────────┘
               │
     ┌─────────▼──────────┐
     │   Qdrant Cloud     │
     │  Hybrid Search     │
     │  ├─ Dense recall   │
     │  ├─ Sparse recall  │
     │  └─ RRF fusion     │
     │  → 25 candidates   │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────┐
     │  FlashRank         │
     │  Cross-Encoder     │
     │  ms-marco-MiniLM   │
     │  → top 5 chunks    │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────┐
     │  Groq LLM          │
     │  llama-3.3-70b     │
     │  Citation prompt   │
     └─────────┬──────────┘
               │
     ┌─────────▼──────────┐
     │  Streaming SSE     │
     │  /ask/stream       │
     └────────────────────┘
```

---

## Files Modified

| File | Change |
|------|--------|
| `app/ingest.py` | pymupdf4llm parsing, SPLADE sparse vectors, UUID IDs, semantic chunking |
| `app/rag_chain.py` | Hybrid search (RRF), FlashRank re-ranking, streaming, model singletons |
| `app/main.py` | `/ask` → `rag_chain.ask_question()`, new `/ask/stream` SSE endpoint |
| `eval/evaluate_v2.py` | Before/after RAGAS evaluation + markdown report generator |
