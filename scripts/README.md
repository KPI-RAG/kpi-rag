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
```

Flags: `--config` (default `configs/config.yaml`), `--rebuild` (empties the collection first and refuses to run when no tickets were loaded). Needs the raw TelecomTS files in `data/raw` (see the top-level README).

### 2) `experiment_retrieval_design.py`
Purpose:
- Compare the legacy template-ticket retrieval with the per-window KPI-narrative retrieval that the system uses.

Needs `data/raw`.

```bash
uv run python scripts/experiment_retrieval_design.py
```

### 3) `run_pipeline.py`
Purpose:
- Run retrieval, alignment-table grounding, LLM generation and citation validation on one `ClassifierOutput` JSON file.

```bash
uv run python scripts/run_pipeline.py --input path/to/classifier_output.json --output explanation.json
```

Flags: `--input` (required), `--output` (required), `--config`.

### 4) `run_eval_track_a.py`
Purpose:
- Recompute detection, classification and retrieval metrics from the committed Layer 2 output.

```bash
uv run python scripts/run_eval_track_a.py
```

Flags: `--output` (default `data/processed/track_a_metrics.json`), `--config`, `--no-retrieval` (skip the ChromaDB queries).

### 5) `run_eval_track_b.py`
Purpose:
- Score Track C condition-C3 explanations with a G-Eval LLM judge, or re-aggregate pre-scored records.

```bash
# Judge mode (needs GROQ_API_KEY, or --api-key and --base-url)
uv run python scripts/run_eval_track_b.py \
  --explanations data/processed/track_c_v3/track_c_explanations.jsonl \
  --alignment data/alignment_table.json \
  --output data/processed/track_b_geval_results.json

# Aggregation-only mode
uv run python scripts/run_eval_track_b.py --scores <scored.jsonl> --output <file>
```

Other flags: `--model` (default `qwen/qwen3.8-27b`), `--human-scores`, `--timeout`, `--workers`, `--delay`.

### 6) `run_eval_track_c.py`
Purpose:
- Run the three-condition citation ablation (C1, C2, C3) on three test windows per true fault type, then score it.

```bash
uv run python scripts/run_eval_track_c.py --summarise-only                    # re-score the stored run, no LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/my_run      # new run, 72 LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/my_run --dry-run
```

Flags: `--output` (default `data/processed/track_c_v3`), `--config`, `--n-per-fault`, `--dry-run`, `--force`, `--summarise-only`. A run never overwrites existing results unless `--force` is given.

### 7) `run_eval_track_d_(teleqna).py`
Purpose:
- Measure retrieval coverage and answer accuracy with and without retrieved context on 600 TeleQnA questions.

```bash
uv run python "scripts/run_eval_track_d_(teleqna).py" --config configs/config.yaml --audit-only
uv run python "scripts/run_eval_track_d_(teleqna).py" --config configs/config.yaml --output data/processed/track_d_full.json
```

Flags: `--config` (required), `--output`, `--model` (default `gemini-3.5-flash-lite`), `--base-url`, `--api-key`, `--max-per-category` (default 200), `--top-k` (default 5), `--delay`, `--audit-only`.

### Standards inspection
`Standards_inspection/` holds the scripts that search the 3GPP knowledge graph. See its [README](Standards_inspection/README.md).
