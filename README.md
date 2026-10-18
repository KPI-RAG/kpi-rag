# KPI-RAG: Explainable 5G Fault Diagnosis

Detects 5G network faults and explains each one: root cause, KPI evidence, the governing 3GPP
standard (checked against an alignment table) and a recommended action. Combines SHAP feature
attribution, retrieval of comparable past incidents and an LLM grounded in 3GPP standards.

## Using the dashboard
1. Choose an input in the sidebar: **Use example**, **Browse Dataset Windows** (the 382 test
   windows Phase 2 flagged as faults; filter by fault type, then pick a window) or **Upload JSON**
   (a `ClassifierOutput` payload).
2. Click **Generate Explanation**.
3. Read the diagnosis, its citation check (✅/❌), the retrieved past incidents and the RCA evidence.

## Architecture

```
Layer 2 (offline, phase2/ on the team main branch): CatBoost cascade + TreeSHAP
        -> data/processed/layer2_output_sessionsplit_phase2.json (6,312 test windows)
                 ↓
ChromaDB retrieval: past incidents of the PREDICTED fault type (metadata filter),
        ranked by KPI-evidence similarity to the window (query carries no label)
                 ↓
Alignment table v1.2 (fault → 3GPP clause; the only source of standards in the prompt)
        + RCA evidence (per window: SHAP features, observed KPI means, causal mechanism)
                 ↓
LLM (Gemini / Groq / Ollama) → structured explanation
                 ↓
Citation validation (TS/TR XX.XXX format + the fault's alignment row)
                 ↓
Streamlit dashboard (6 panels)
```

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Gemini API key](https://aistudio.google.com/apikey) (free tier works; Groq is an optional fallback backend)

### 1. Clone and install

The system lives on the **`master`** branch.

```bash
git clone -b master https://github.com/KPI-RAG/kpi-rag.git
cd kpi-rag
uv sync            # CPU-only torch; no PYTHONPATH needed afterwards
```

Behind an HTTPS-inspecting antivirus or proxy (`CERTIFICATE_VERIFY_FAILED`): add `--system-certs`
to `uv` commands, and run LLM scripts with the Windows trust store, e.g.
`uv run --system-certs --with truststore python -c "import truststore; truststore.inject_into_ssl(); import runpy, sys; sys.argv=['scripts/run_eval_track_c.py', '--output', 'data/processed/my_run']; runpy.run_path(sys.argv[0], run_name='__main__')"`

### 2. API key

```bash
cp .env.example .env      # then put your GEMINI_API_KEY in .env
```

### 3. Raw data (only needed to rebuild the index or rerun the retrieval experiment)

```bash
uv run hf download AliMaatouk/TelecomTS --repo-type dataset --local-dir data/raw   # 33 JSONL files, ~1.2 GB
```

Keep the files where the download puts them: ticket ids are positions in sorted file order,
which `build_index.py` checks against the ground truth before indexing.

### 4. Build the ChromaDB index (already committed in `data/chroma_db/`)

```bash
uv run python scripts/build_index.py --rebuild
```

Indexes the 935 anomalous **training** windows (one document per window: its KPI narrative).
The 300 anomalous test windows are never indexed. `--rebuild` empties the collection first and
refuses to run when no tickets were loaded, so a missing `data/raw` cannot wipe the index.

### 5. Run the pipeline (single explanation)

```bash
uv run python scripts/run_pipeline.py --input path/to/classifier_output.json --output explanation.json
```

### 6. Run the dashboard

```bash
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

**Streamlit Community Cloud:** deploy from branch `master`, main file `dashboard/app.py`, Python 3.11,
and set `GEMINI_API_KEY` under *Secrets*. Cloud installs from `uv.lock` (CPU torch); the app needs only
committed files.

## Evaluation

### Track A — detection, classification, retrieval

```bash
uv run python scripts/run_eval_track_a.py                 # -> data/processed/track_a_metrics.json
uv run python scripts/experiment_retrieval_design.py      # retrieval design comparison (needs data/raw)
```

Recomputed from the Phase 2 predictions and ground truth (matches Farida's `validation.json`):
binary recall **266/300 = 88.7 %**, precision **266/382 = 69.6 %**, exact 11-class accuracy
**199/300 = 66.3 %** (74.8 % of detected faults; **51.5 % without Jamming**, which is 94 of the 300
test faults), congestion-family accuracy 75.0 %. Retrieval diagnostic: with the label removed, the
retrieved incidents' class matches the true class at P@1 = 0.451 (majority-class share 0.353) — text
cannot identify the fault, which is why the class is applied as a filter.

### Track B — G-Eval (LLM judge, Rodina Khallaf)

```bash
uv run python scripts/run_eval_track_b.py --scores <scored.jsonl> --output data/processed/track_b_results.json
```

Pending: `data/processed/track_c_v3/track_c_scores_template.jsonl` holds the 72 explanations to score
(1–5 on four dimensions, citation judged against the true fault's alignment row). Rows without
`reference_valid`, or still at 0.0, are rejected; extra keys such as a judge's rationale are ignored.

### Track C — citation ablation (3 conditions)

Sampling is by **ground truth**: 3 test windows per true fault type (10 types, Jamming excluded),
each run through what the system actually does (Phase 2 prediction → retrieval → LLM). All three
conditions use the same instructions; they differ only in context. Results of run v3
(`data/processed/track_c_v3/`; run v2 gave the same counts before the evidence-block fixes):

| Answer rule | Correct vs true fault (n=30) | When classified correctly (n=19) |
|-------------|-----------------------------|----------------------------------|
| C1 LLM: label + SHAP | 5/30 (17 %) | 4/19 |
| C2 LLM: + 3 past incidents of the predicted type | 7/30 (23 %) | 6/19 |
| C3 LLM: + alignment row + RCA evidence | **21/30 (70 %)** | **19/19** |
| No LLM: always cite TS 38.141-1 | 12/30 (40 %) | — |
| No LLM: copy the predicted fault's alignment row | 21/30 (70 %) | 19/19 |

Gemini 3.5 Flash Lite, `random_state=42`, alignment table v1.2, 0 template fallbacks, 72 LLM calls.

> **How to read it.** Alignment-table grounding is effective: when the correct standard is provided, the
> LLM cites it in 19/19 cases, and C2→C3 adds 47 points. C3 matching the copy-row rule confirms the table
> as the grounding mechanism — the LLM reliably adopts the provided standard. Without the table, citations
> are unreliable (C1, C2 are below a fixed answer). All 9 C3 failures are Layer 2 errors:
> 6 detector misses (all 3 Sudden Spike samples — Model 1 flags none of the 11 Sudden Spike test
> windows) and 3 misclassifications. n = 30 from few recording sessions: treat the counts as descriptive.

```bash
uv run python scripts/run_eval_track_c.py --output data/processed/my_run --dry-run   # 1 per fault (27 LLM calls)
uv run python scripts/run_eval_track_c.py --output data/processed/my_run             # 3 per fault (72 LLM calls)
uv run python scripts/run_eval_track_c.py --output data/processed/track_c_v3 --summarise-only   # re-score, no LLM
```

A run refuses to overwrite an existing one in `--output` unless `--force` is given. Outputs:
`track_c_explanations.jsonl`, `track_c_summary.json` (with the no-LLM baselines),
`track_c_scores_template.jsonl`. The earlier prediction-sampled run (alignment v1.1) is kept in
`data/processed/track_c_explanations.jsonl` for comparison only.

## Testing

```bash
uv run pytest tests/ -q
```

209 tests in 29 modules (21 unit, 8 integration): 206 pass, 3 skip without `GROQ_API_KEY`. The test
session restores `data/chroma_db/` afterwards (ChromaDB rewrites its files whenever it is opened).

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
│   ├── ml_model.py              # CatBoost loader (tests only; Layer 2 runs offline)
│   └── utils.py                 # Logging, 3GPP ref validation
├── scripts/
│   ├── build_index.py           # Index builder CLI
│   ├── run_pipeline.py          # Single-explanation CLI
│   ├── run_eval_track_a.py      # Track A + retrieval metrics CLI
│   ├── experiment_retrieval_design.py  # retrieval design comparison (report §5.5)
│   ├── run_eval_track_b.py      # Track B evaluation CLI
│   └── run_eval_track_c.py      # Track C ablation CLI
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   └── components/              # 6 dashboard panels
├── tests/                       # 21 unit + 8 integration test modules
├── data/
│   ├── raw/                     # TelecomTS JSONL (gitignored; see Quick Start step 3)
│   ├── chroma_db/               # ChromaDB store (committed — Streamlit Cloud has no raw data to rebuild it)
│   ├── models/                  # CatBoost model1.cbm / model2.cbm (Layer 2 runs offline)
│   ├── alignment_table.json     # 11-row fault→3GPP mapping (v1.2; Jamming not-applicable)
│   ├── indices/                 # train/test window indices (session split)
│   └── processed/               # RCA evidence, layer-2 outputs, Track A metrics, track_c_v3/ (v2 kept)
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
