# KPI-RAG: Explainable 5G Fault Diagnosis

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://docs.astral.sh/uv/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-207%20passed%20%7C%203%20skipped-brightgreen.svg)](#testing)

KPI-RAG detects faults in 5G radio access network telemetry and explains each one: the root cause,
the KPI evidence behind it, the responsible O-RAN component, a recommended action and the governing
3GPP specification. The cited specification is checked automatically against a 3GPP Release 19 /
O-RAN alignment table, so an explanation cannot silently cite the wrong standard.

Master's project (CISC 898, Queen's University, Group 15) with industry partner Capgemini,
supervised by Prof. Yuanzhu Chen. Built on the public
[TelecomTS](https://huggingface.co/datasets/AliMaatouk/TelecomTS) dataset.

## Contents

- [Key results](#key-results)
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Reproducing the evaluation](#reproducing-the-evaluation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Key results

Measured on five held-out recording sessions (6,312 test windows, 300 true faults).

| What | Result |
|---|---|
| Fault detection (Model 1) | 116 false alarms (first prototype: 1,105); precision 0.70, recall 0.887, F1 0.78 |
| Fault type (cascade) | 74.8 % exact accuracy on detected faults; 51.5 % excluding Jamming |
| Citation correctness, no alignment table (C1/C2) | 5/30 |
| Citation correctness, with alignment table (C3) | **21/30 (70 %)**, and **19/19** when the fault type is right |
| Explanation quality (LLM judge, 30 explanations) | 3.59 / 5 |
| TeleQnA with retrieval confidence gate | 76.0 % (no-retrieval baseline 76.3 %) |

All 9 remaining C3 errors start upstream of the explanation layer (6 detector misses, 3
misclassifications): once the fault is identified, the cited standard is correct.

## How it works

```
TelecomTS window (18 KPI channels x 128 steps)
        |
        v
Layer 2 (offline): CatBoost cascade + TreeSHAP
        Model 1: fault / normal      Model 2: fault type (11 classes)
        -> data/processed/layer2_output_sessionsplit_phase2.json
        |
        v
Retrieval: past incidents of the PREDICTED fault type (ChromaDB metadata filter),
        ranked by KPI-evidence similarity; the query carries no label
        |
        v
Prompt: SHAP evidence + incidents + the fault's alignment-table row (v1.2)
        + per-window RCA evidence (top SHAP features, observed KPI means, causal mechanism)
        |
        v
LLM (Gemini; Groq or Ollama as alternatives) -> structured JSON explanation
        |
        v
Citation validation: TS/TR number format + membership in the fault's alignment row
        |
        v
Streamlit dashboard (6 panels)
```

The alignment table is the only source of standards in the prompt. Retrieval supplies concrete KPI
patterns from comparable incidents; it does not decide the fault type, because the dataset's tickets
are one text template per class.

## Getting started

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A [Gemini API key](https://aistudio.google.com/apikey) (the free tier works)

### Install

```bash
git clone -b master https://github.com/KPI-RAG/kpi-rag.git
cd kpi-rag
uv sync                      # installs CPU-only torch; no PYTHONPATH needed
cp .env.example .env         # then set GEMINI_API_KEY in .env
```

The trained models (`data/models/`), the Layer 2 output, the RCA evidence and the ChromaDB index
(`data/chroma_db/`) are committed, so the dashboard and the evaluation scripts run without the raw
dataset.

<details>
<summary>Behind an HTTPS-inspecting proxy or antivirus (<code>CERTIFICATE_VERIFY_FAILED</code>)</summary>

Add `--system-certs` to `uv` commands and run LLM scripts with the operating system's trust store:

```bash
uv run --system-certs --with truststore python -c "import truststore; truststore.inject_into_ssl(); import runpy, sys; sys.argv=['scripts/run_eval_track_c.py', '--output', 'data/processed/my_run'...]
```
</details>

### Raw data (optional)

Needed only to rebuild the index or rerun the retrieval experiment:

```bash
uv run hf download AliMaatouk/TelecomTS --repo-type dataset --local-dir data/raw   # 33 JSONL files, ~1.2 GB
uv run python scripts/build_index.py --rebuild
```

Keep the files where the download puts them: ticket ids are positions in sorted file order, and
`build_index.py` checks every ticket against the ground truth before indexing. The index holds the
935 anomalous training windows only; `--rebuild` empties the collection first and refuses to run when
no tickets were loaded.

## Usage

### Dashboard

```bash
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

1. Choose an input in the sidebar: **Use example**, **Browse Dataset Windows** (the 382 test windows
   flagged as faults, filterable by fault type) or **Upload JSON** (a `ClassifierOutput` payload).
2. Click **Generate Explanation**.
3. Read the diagnosis, its citation check, the retrieved past incidents and the RCA evidence.

To deploy on Streamlit Community Cloud: branch `master`, main file `dashboard/app.py`, Python 3.11,
and `GEMINI_API_KEY` under *Secrets*.

### Command line

```bash
uv run python scripts/run_pipeline.py --input path/to/classifier_output.json --output explanation.json
```

## Reproducing the evaluation

| Track | Measures | Command | Output |
|---|---|---|---|
| A | Detection, classification, retrieval | `uv run python scripts/run_eval_track_a.py` | `data/processed/track_a_metrics.json` |
| A | Retrieval design comparison (needs raw data) | `uv run python scripts/experiment_retrieval_design.py` | printed |
| B | Explanation quality (G-Eval LLM judge) | `uv run python scripts/run_eval_track_b.py --scores <scored.jsonl> --output <file>` | JSON |
| C | Citation ablation (3 prompt conditions) | see below | `data/processed/track_c_v3/` |

### Track C

Three test windows per true fault type (10 types; Jamming has no applicable 3GPP clause), seed 42,
each run through the full system. The conditions share one instruction block and differ only in
context: C1 gets the label and SHAP evidence, C2 adds three past incidents, C3 adds the alignment
row and the window's RCA evidence. A citation counts as correct only against the **true** fault's
alignment row; detector misses count as failures.

| Answer rule | Correct (n = 30) | Fault type right (n = 19) |
|---|---|---|
| C1 LLM: label + SHAP | 5/30 | 4/19 |
| C2 LLM: + past incidents | 5/30 | 4/19 |
| C3 LLM: + alignment row + RCA evidence | **21/30** | **19/19** |
| No LLM: always cite TS 38.141-1 | 12/30 | — |
| No LLM: copy the predicted fault's row | 21/30 | 19/19 |

C3 equals the copy-row rule by design: the LLM adopts the standard the table provides, and the
validator holds it to that row. With n = 30 from a few recording sessions, treat the counts as
descriptive.

```bash
uv run python scripts/run_eval_track_c.py --summarise-only                    # re-score the stored run, no LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/my_run      # new run, 72 LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/my_run --dry-run   # 1 window per fault
```

A run never overwrites existing results unless `--force` is given. Outputs:
`track_c_explanations.jsonl`, `track_c_summary.json` (with the no-LLM baselines) and
`track_c_scores_template.jsonl` (input for the Track B judge).

## Testing

```bash
uv run pytest tests/ -q
```

210 tests in 29 modules (21 unit, 8 integration): 207 pass and 3 are skipped without
`GROQ_API_KEY`. The test session restores `data/chroma_db/` afterwards, because ChromaDB rewrites
its files whenever it is opened.

## Configuration

All settings live in [`configs/config.yaml`](configs/config.yaml); API keys come from `.env`
(see [`.env.example`](.env.example)).

| Setting | Value | Notes |
|---|---|---|
| `rag.embedding_model` | `all-MiniLM-L6-v2` | Sentence-BERT, 384 dimensions |
| `rag.collection_name` | `telecomts_descriptions` | One document per training window (its KPI narrative) |
| `rag.top_k` | `5` | Past incidents per query, filtered to the predicted fault type |
| `rag.cosine_threshold` | `0.35` | Low-confidence flag; test top-1 similarity is 0.59 to 0.75, so it never fired |
| `llm.backend` | `gemini` | Also `groq`, `ollama` |
| `llm.gemini_model` | `gemini-3.5-flash-lite` | Free tier: 15 requests per minute |
| `llm.max_retries` | `3` | Backs off on rate limits (server `Retry-After`, else exponential with jitter); config and auth errors stop at once; template fallback at the end |
| `llm.min_request_interval_s` | `2.0` | Minimum gap between LLM calls |

## Repository layout

```
.
├── configs/config.yaml          # retrieval, LLM, data and evaluation settings
├── src/                         # library code
│   ├── schema.py                #   Pydantic models (11 fault types)
│   ├── data_loader.py           #   TelecomTS loader and train/test split
│   ├── kg_indexer.py            #   ChromaDB indexing
│   ├── rag_query.py             #   retrieval with fault-type filter
│   ├── rca_loader.py            #   per-window RCA evidence for the prompt
│   ├── rca_utils.py             #   KPI evidence assessment and RCA utilities
│   ├── llm_explainer.py         #   prompts, LLM backends, retries, citation validation
│   ├── evaluator.py             #   Track B and C metrics
│   └── ml_model.py, utils.py, config_loader.py
├── scripts/                     # command-line entry points (index, pipeline, Tracks A to C)
├── dashboard/                   # Streamlit app and its 6 panels
├── tests/                       # 21 unit and 8 integration test modules
├── data/
│   ├── alignment_table.json     # fault -> 3GPP/O-RAN mapping, v1.2
│   ├── models/                  # CatBoost Model 1 and Model 2
│   ├── indices/                 # session-split train/test indices
│   ├── chroma_db/               # retrieval index (committed for Streamlit Cloud)
│   └── processed/               # Layer 2 output, RCA evidence, Track A metrics, Track C run
├── pyproject.toml, uv.lock      # environment (uv)
└── requirements.txt             # pinned export of uv.lock
```

The Layer 2 training pipeline (feature engineering, model experiments) lives in `phase2/` on the
`main` branch; this branch consumes its outputs.

## Team

| Member | Main contributions |
|---|---|
| Ahmed Mohamed Al Shobaki (team leader) | Retrieval and its redesign, prompt design, LLM explanation layer and citation validation, Track C, dashboard, deployment, test suite |
| Farida El Ghabary | Synthetic training data and the synthetic-data model |
| Raneem Alnaghy | Recording-session reconstruction and the session-level split; false-alarm filter study |
| Rodina Khallaf | 3GPP Release 19 / O-RAN alignment table; Track B (G-Eval) and Track D (TeleQnA) evaluations |

## Acknowledgements

Prof. Yuanzhu Chen (supervisor); Capgemini (industry partner); Queen's University and its Frontenac
cluster; the National Telecommunication Institute (NTI) and the Ministry of Communications and
Information Technology of Egypt (DiGiLiANS Master Program). The TelecomTS dataset is by Feng et al.,
"TelecomTS: A multi-modal observability dataset for time series and language analysis",
[arXiv:2510.06063](https://arxiv.org/abs/2510.06063), 2025.

## License

Academic coursework for Queen's University (CISC 898, Group 15). No open-source license is granted;
contact the authors before reusing the code. The TelecomTS dataset is subject to its own license on
Hugging Face.
