# RAG Evaluation Framework (GDPR) — Technical Report

## 1. Goal
Build a reproducible evaluation framework for a RAG system, focusing on grounding, correctness, and reliability.

## 2. System Under Test
- Vector store: ChromaDB (persistent)
- Embeddings: text-embedding-3-small
- Generator model: gpt-4.1-mini
- Data source: GDPR (CELEX_32016R0679_EN_TXT.pdf)

## 3. Evaluation Methodology
### 3.1 CI Golden Suite (Frozen Context)
- Dataset: evaluation/datasets/ci_golden.json
- Quality gate: mean overall >= 0.80

### 3.2 End-to-End Benchmark (Nightly Mode)
- Retrieval top_k = 4
- Answer requires citations [i] pointing to retrieved excerpts

### 3.3 LLM-as-a-Judge
- Outputs strict JSON rubric scores:
  relevance, correctness, grounding, completeness, reasoning_quality, overall

### 3.4 Reliability
- Repeated judge scoring (N=5)
- Report mean/std; stability gate std <= 0.10

## 4. Results (current)
- CI mean overall: 0.91
- Nightly mean overall: 0.95
- Judge reliability std: 0.064

## 5. Ablations
Summarize results from evaluation/artifacts/ablation_results.json:
- Effect of chunk_size
- Effect of top_k

## 6. Known Failure Modes
- Retrieval misses the most relevant article/definition
- Model answers using partially-related excerpt (semantic drift)
- Over-citation (too many chunks) vs under-citation

## 7. Next Steps
- Add retrieval quality metrics (hit@k, context precision)
- Add adversarial / unanswerable questions to test hallucination resistance
- Add citation coverage metric (each claim must map to at least one excerpt)
