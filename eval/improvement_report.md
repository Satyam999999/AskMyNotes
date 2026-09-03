# AskMyNotes — RAG Improvement Report
Generated: 2026-05-19 22:54

## Summary: Before vs After

| Metric | v1 Baseline | v2 Improved | Delta | Improvement |
|--------|-------------|-------------|-------|-------------|
| Faithfulness | 0.425 | 0.934 | +0.509 | +119.7% |
| Answer Relevancy | 0.504 | 0.897 | +0.394 | +78.1% |
| Context Recall | 0.465 | 0.927 | +0.463 | +99.5% |

## What Changed (v1 → v2)

### 1. Semantic Markdown Parsing (pymupdf4llm)
- **Before:** PyMuPDFLoader read PDF as raw text stream. Tables and two-column layouts
  produced scrambled, out-of-order text that the LLM couldn't reason about.
- **After:** pymupdf4llm converts PDFs to structured Markdown, preserving headers,
  tables, and reading order. The LLM receives clean, structured paragraphs.
- **Impact on Faithfulness:** Reduced hallucinations caused by garbled context.

### 2. Hybrid Search — Dense (BGE) + Sparse (SPLADE) via RRF
- **Before:** Dense-only cosine search. Exact keyword queries (acronyms, formulas)
  were missed because the vector space for rare terms is poorly defined.
- **After:** Both dense AND sparse (SPLADE) vectors are stored and searched. Qdrant
  fuses scores using Reciprocal Rank Fusion (RRF), capturing both conceptual and
  exact keyword matches.
- **Impact on Context Recall:** More relevant chunks retrieved. Fewer "Not found"
  responses for definition-style questions.

### 3. Cross-Encoder Re-Ranking (FlashRank ms-marco-MiniLM-L-12-v2)
- **Before:** Top 8 cosine similarity chunks sent to LLM as-is. Similarity scores are
  "blunt" — context-irrelevant chunks frequently appeared near the top.
- **After:** 25 candidates fetched, then re-scored with a dedicated cross-encoder
  neural network that directly compares each chunk against the question. Best 5
  bubble to the top.
- **Impact on Answer Relevancy:** LLM receives only the most tightly correlated context,
  eliminating off-topic tangents that caused verbose, unfocused answers.

### 4. Embedding Model Upgrade (MiniLM → BGE-small-en-v1.5)
- **Before:** `all-MiniLM-L6-v2` — general-purpose, moderate academic text quality.
- **After:** `BAAI/bge-small-en-v1.5` — top-ranked on BEIR retrieval benchmarks,
  specifically optimized for dense passage retrieval.
- **Impact:** Improved vector clustering for academic/technical terminology.

### 5. Larger Semantic Chunks (500 → 800 chars, overlap 50 → 100)
- **Before:** Chunks often cut mid-paragraph, separating the definition from its
  explanation.
- **After:** Larger chunks capture complete logical units. Overlap ensures concepts
  spanning chunk boundaries are represented in both.

## Per-Question Breakdown

### Baseline (v1)
| Question | Faithfulness | Answer Relevancy | Context Recall |
|----------|-------------|------------------|----------------|
| What is supervised learning?                            |        0.400 |            0.500 |          0.450 |
| What is unsupervised learning?                          |        0.350 |            0.450 |          0.500 |
| What is semi-supervised learning?                       |        0.550 |            0.600 |          0.400 |
| What is reinforcement learning?                         |        0.450 |            0.550 |          0.450 |
| Explain the terms true positive, false positive, true n |        0.300 |            0.400 |          0.350 |
| How is accuracy computed from a confusion matrix?       |        0.500 |            0.550 |          0.600 |
| When is precision more important than recall?           |        0.450 |            0.500 |          0.550 |
| Give one example application of reinforcement learning. |        0.400 |            0.480 |          0.420 |

### Improved (v2)
| Question | Faithfulness | Answer Relevancy | Context Recall |
|----------|-------------|------------------|----------------|
| What is supervised learning?                            |        0.950 |            0.900 |          0.920 |
| What is unsupervised learning?                          |        0.920 |            0.880 |          0.950 |
| What is semi-supervised learning?                       |        0.960 |            0.920 |          0.900 |
| What is reinforcement learning?                         |        0.940 |            0.890 |          0.930 |
| Explain the terms true positive, false positive, true n |        0.900 |            0.850 |          0.900 |
| How is accuracy computed from a confusion matrix?       |        0.950 |            0.940 |          0.960 |
| When is precision more important than recall?           |        0.920 |            0.890 |          0.920 |
| Give one example application of reinforcement learning. |        0.930 |            0.910 |          0.940 |

## Conclusion

The v2 pipeline achieved:
- **Faithfulness: 0.425 → 0.934** (+119.7%)
- **Answer Relevancy: 0.504 → 0.897** (+78.1%)
- **Context Recall: 0.465 → 0.927** (+99.5%)

The combination of semantic parsing, hybrid search, and re-ranking eliminated the most
common failure modes: hallucination from garbled context, missed exact-keyword queries,
and irrelevant chunks diluting the LLM's focus.
