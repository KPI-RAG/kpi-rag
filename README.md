---
title: KPI-RAG 5G Fault Diagnosis
emoji: 📡
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.35.0
app_file: dashboard/app.py
pinned: false
license: mit
---

# KPI-RAG: Explainable 5G Fault Diagnosis

RAG-augmented LLM explanations with 3GPP alignment.

## What it does
Detects 5G network faults and generates structured explanations
with root cause, KPI evidence, 3GPP standards reference,
and recommended action.

## How to use
1. Select a fault type from the sidebar
2. Choose a window index
3. Click "Generate Explanation"
4. View the structured diagnosis with 3GPP citation

An explainability framework for 5G network fault classification that combines SHAP-based feature attribution, knowledge-graph retrieval, and LLM-generated explanations grounded in 3GPP standards.

## Architecture

```
Layer 2 (offline, phase2/ on the team main branch): CatBoost cascade + TreeSHAP
        -> data/processed/layer2_output_sessionsplit_phase2.json (6,312 test windows)
                 ↓
ChromaDB retrieval: past incidents of the PREDICTED fault type (metadata filter),
        ranked by KPI-evidence similarity to the window (query carries no label)
                 ↓
Alignment table v1.2 (fault → 3GPP clause) + RCA evidence (per window)
                 ↓
LLM (Gemini / Groq / Ollama) → structured explanation
                 ↓
Citation validation (TS/TR XX.XXX format + the fault's alignment row)
                 ↓
Streamlit dashboard (6 panels)
```

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- [Gemini API key](https://aistudio.google.com/apikey) (free tier works; Groq is an optional fallback backend)

### 1. Clone and install

```bash
git clone <repo-url>
cd project
uv sync
```

### 2. Set environment variables

```bash
# PowerShell
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Or set directly:
$env:GEMINI_API_KEY="..."
$env:PYTHONPATH="."
```

Behind an HTTPS-inspecting antivirus or proxy (Python reports `CERTIFICATE_VERIFY_FAILED`),
run LLM scripts with the Windows trust store:
`uv run --system-certs --with truststore python -c "import truststore; truststore.inject_into_ssl(); import runpy, sys; sys.argv=['scripts/run_eval_track_c.py']; runpy.run_path(sys.argv[0], run_name='__main__')"`

### 3. Place raw data

Download the TelecomTS dataset and place JSONL files under:

```
data/raw/          # 33 JSONL files (nested structure supported)
```

### 4. Build the ChromaDB index

```bash
uv run python scripts/build_index.py --rebuild
```

Indexes the 935 anomalous **training** windows (one document per window: its KPI narrative)
into `data/chroma_db/`. The 300 anomalous test windows are never indexed. `--rebuild` empties
the collection first; without it existing ids are skipped, so a stale index would survive.

### 5. Run the pipeline (single explanation)

```bash
uv run python scripts/run_pipeline.py --input path/to/classifier_output.json --output explanation.json
```

### 6. Run the dashboard

```bash
$env:PYTHONPATH="."
$env:GEMINI_API_KEY="..."
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

"Browse Dataset Windows" lists the 382 test windows Phase 2 flagged as faults.

## Evaluation

### Track A — detection, classification, retrieval

```bash
uv run python scripts/run_eval_track_a.py      # -> data/processed/track_a_metrics.json
```

Recomputed from the Phase 2 predictions and ground truth (matches Farida's `validation.json`):
binary recall **266/300 = 88.7 %**, precision **266/382 = 69.6 %**, exact 11-class accuracy
**199/300 = 66.3 %** (74.8 % of detected faults), congestion-family accuracy 75.0 %.
Retrieval diagnostic: with the label removed, the retrieved incidents' class matches the true
class at P@1 = 0.451 (majority-class share 0.353) — text cannot identify the fault, which is
why the class is applied as a filter.

### Track B — human G-Eval

```bash
uv run python scripts/run_eval_track_b.py --scores <rated_scores.jsonl> --output data/processed/track_b_results.json
```

Not done yet: `data/processed/track_c_v2/track_c_scores_template.jsonl` holds the 72 explanations
to rate (1–5 on four dimensions). Un-rated rows (0.0) are rejected, so the template cannot be
averaged by mistake.

### Track C — citation ablation (3 conditions)

Sampling is by **ground truth**: 3 test windows per true fault type (10 types, Jamming excluded),
each run through what the system actually does (Phase 2 prediction → retrieval → LLM). All three
conditions use the same instructions; they differ only in context.

| Condition | Context | Correct vs true fault (n=30) | When classified correctly (n=19) |
|-----------|---------|-----------------------------|----------------------------------|
| C1 | Label + SHAP | 5/30 (17 %) | 4/19 (21 %) |
| C2 | + 3 past incidents of the predicted type | 7/30 (23 %) | 6/19 (32 %) |
| C3 | + alignment row + RCA evidence | **21/30 (70 %)** | **19/19 (100 %)** |

Gemini 3.5 Flash Lite, `random_state=42`, alignment table v1.2, 0 template fallbacks, 72 LLM calls.
C2→C3: 14 windows improve, 0 get worse (exact McNemar p = 1.2·10⁻⁴). Wilson 95 % CI for C3:
52–83 % end to end, 83–100 % when classified correctly.

All 9 C3 failures come from Layer 2, not from grounding: 6 windows the detector missed (all 3
Sudden Spike samples — Model 1 flags none of the 11 Sudden Spike test windows) and 3 misclassified.

> **What C3 measures.** When the classifier is right, the correct standard is in the C3 prompt,
> so 19/19 is *prompt compliance*. The end-to-end figure (21/30) is not circular: it scores the
> citation against the true fault, so a misclassification that injects the wrong row counts as wrong.

```bash
uv run python scripts/run_eval_track_c.py --output data/processed/track_c_v2 --dry-run   # 1 per fault
uv run python scripts/run_eval_track_c.py --output data/processed/track_c_v2             # 3 per fault
```

Outputs (`data/processed/track_c_v2/`): `track_c_explanations.jsonl`, `track_c_summary.json`,
`track_c_scores_template.jsonl`. The earlier prediction-sampled run (alignment v1.1) is kept in
`data/processed/track_c_explanations.jsonl` for comparison only.

## Testing

```bash
uv run pytest tests/ -q
```

182 tests in 28 modules (20 unit, 8 integration): 179 pass, 3 skip without `GROQ_API_KEY`.

## Project Structure

```
project/
├── configs/
│   └── config.yaml              # RAG, LLM, data, eval settings
├── src/
│   ├── schema.py                # Pydantic models (11 fault types)
│   ├── config_loader.py         # YAML config loader
│   ├── data_loader.py           # TelecomTS JSONL loader
│   ├── kg_indexer.py            # ChromaDB indexing
│   ├── rag_query.py             # Cosine similarity retrieval
│   ├── rca_loader.py            # RCA evidence lookup by window_index
│   ├── llm_explainer.py         # Gemini/Groq/Ollama LLM, 3-condition prompts, citation validation
│   ├── evaluator.py             # G-Eval scoring + Track B/C metrics
│   └── utils.py                 # Logging, 3GPP ref validation
├── scripts/
│   ├── build_index.py           # Index builder CLI
│   ├── run_pipeline.py          # Single-explanation CLI
│   ├── run_eval_track_a.py      # Track A + retrieval metrics CLI
│   ├── run_eval_track_b.py      # Track B evaluation CLI
│   └── run_eval_track_c.py      # Track C ablation CLI
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   └── components/              # 6 dashboard panels
├── tests/                       # 20 unit + 8 integration test modules
├── data/
│   ├── raw/                     # TelecomTS JSONL (gitignored)
│   ├── indices/                 # train/test window indices (gitignored)
│   ├── chroma_db/               # ChromaDB store (committed — Streamlit Cloud has no raw data to rebuild it)
│   ├── models/                  # CatBoost model1.cbm / model2.cbm (Layer 2 runs offline)
│   ├── alignment_table.json     # 11-row fault→3GPP mapping (v1.2; Jamming not-applicable)
│   └── processed/               # RCA evidence, layer-2 outputs, Track A metrics, track_c_v2/
├── docs/
│   ├── KPI-RAG_Proposal_Group_15.pdf
│   ├── KPI_RAG_Final_Report.md      # final report, every number traced to a file
│   ├── REVIEW_2026-10-06.md     # first end-to-end review
│   └── REVIEW_2026-10-18.md     # second review + how each finding was resolved
├── pyproject.toml / uv.lock     # dev environment (uv)
├── requirements.txt             # pinned install for Streamlit Cloud
└── .env.example
```

## Configuration

Key settings in `configs/config.yaml`:

| Setting | Value | Notes |
|---------|-------|-------|
| `rag.embedding_model` | `all-MiniLM-L6-v2` | Sentence-BERT |
| `rag.collection_name` | `telecomts_descriptions` | One document per training window (its KPI narrative) |
| `rag.cosine_threshold` | `0.35` | Low-confidence flag only; test top-1 similarity is 0.59–0.75, so it never fired (0/382) |
| `rag.top_k` | `5` | Past incidents per query, filtered to the predicted fault type |
| `llm.backend` | `gemini` | Also supports `groq`, `ollama` |
| `llm.gemini_model` | `gemini-3.5-flash-lite` | Free tier: 15 requests/min |
| `llm.groq_model` | `llama-3.1-8b-instant` | Fallback backend |
| `llm.max_retries` | `3` | Waits on 429s (server Retry-After, else exponential backoff); config/auth errors stop at once; every failure is logged; template fallback at the end |
| `llm.min_request_interval_s` | `2.0` | Minimum gap between LLM calls (all backends) |

## Fault Types

10 fault types evaluated (Jamming excluded — no applicable 3GPP clause):

1. Co-Channel Interference (Mild / Severe)
2. Buffer Overflow (Gradual Buildup)
3. Antenna Failure
4. Faulty RF Filters (Temporal)
5. High Network Congestion (Gradual / Sudden)
6. Doppler Shift (Severe)
7. Faulty Handover Algorithm (Too Frequent)
8. Resource Allocation Bugs

## License

Academic project — Queen's University MSc Data Science, Group 15.
