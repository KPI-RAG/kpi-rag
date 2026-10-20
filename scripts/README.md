# Scripts

This directory contains the operational Python entry points for the KPI-RAG workflow. The scripts cover data indexing, retrieval experiments, end-to-end pipeline execution, and evaluation tracks A-D.

Use these scripts from the repository root unless otherwise noted.

## Directory overview

- `build_index.py` — build or refresh the ChromaDB retrieval index from ticket data.
- `experiment_retrieval_design.py` — benchmark retrieval design choices and document/query formulations.
- `run_pipeline.py` — run the end-to-end inference pipeline on a single classifier output.
- `run_eval_track_a.py` — recompute Track A detection/classification and retrieval metrics.
- `run_eval_track_b.py` — score generated explanations with a G-Eval judge.
- `run_eval_track_c.py` — generate explanation variants across citation ablation conditions.
- `run_eval_track_d_(teleqna).py` — evaluate KPI-RAG retrieval coverage and answer quality on TeleQnA.

## Script details

### 1) `build_index.py`
Purpose:
- Load raw telecom incident data.
- Filter anomalous tickets.
- Apply the training split.
- Build or refresh the vector index used by the retrieval layer.

Typical usage:
```bash
python scripts/build_index.py --config configs/config.yaml
python scripts/build_index.py --config configs/config.yaml --rebuild