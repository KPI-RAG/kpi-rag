# KPI-RAG: Explainable Root-Cause Analysis for 5G Networks

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://docs.astral.sh/uv/)
[![CatBoost](https://img.shields.io/badge/ML-CatBoost%20%2B%20TreeSHAP-yellow.svg)](https://catboost.ai/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kpi-rag-group15.streamlit.app/)
[![Tests](https://img.shields.io/badge/tests-207%20passed%20%7C%203%20skipped-brightgreen.svg)](#testing)

KPI-RAG detects faults in 5G radio access network (RAN) telemetry and explains each one: the root
cause, the KPI evidence behind it, the responsible O-RAN component, a recommended action and the
governing 3GPP specification. Every cited specification is checked automatically against a
3GPP Release 19 / O-RAN alignment table, so an explanation cannot silently cite the wrong standard.

The system has three layers: a CatBoost detector and fault classifier with TreeSHAP attributions
(Layers 1–2), and a retrieval-augmented explanation layer with citation validation and a Streamlit
dashboard (Layer 3). It is built on the public
[TelecomTS](https://huggingface.co/datasets/AliMaatouk/TelecomTS) dataset.

This is a Master's project (CISC 898, Queen's University, Group 15) carried out with industry
partner Capgemini and supervised by Prof. Yuanzhu Chen. The full write-up is in the final report
(`KPI_RAG_Final_Report`).

## Table of contents

- [Key results](#key-results)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Key results

Detection and classification are measured on five held-out recording sessions (sessions 1, 3, 4,
13 and 17): 6,312 test windows, of which 300 are true faults. Sources are the committed JSON files
in [`data/processed/`](data/processed/).

| Area | Metric | First prototype | Final system | Source |
|---|---|---|---|---|
| Detection (Model 1) | False alarms on 6,012 normal windows | 1,105 | **116** | [`validation.json`](data/processed/validation.json) |
| Detection (Model 1) | Precision / recall / F1 | 0.20 / 0.90 / 0.32 | **0.70 / 0.89 / 0.78** | [`track_a_metrics.json`](data/processed/track_a_metrics.json), [`Data_and_Models/README.md`](Data_and_Models/README.md) |
| Classification (Model 2) | Cross-validated macro-F1 (25 session folds) | 0.291 | **0.389** | [`phase2_results.md`](Data_and_Models/3_Final_system/docs/phase2_results.md) |
| Classification (Model 2) | Test accuracy / macro-F1 on the 300 true faults | 0.677 / 0.537 | **0.703 / 0.591** | [`phase2_results.md`](Data_and_Models/3_Final_system/docs/phase2_results.md) |
| Cascade | Windows passed to the explanation layer | 1,374 (1,105 false) | **382 (266 true, 116 false)** | [`validation.json`](data/processed/validation.json) |
| Cascade | Exact fault-type accuracy | | 74.8 % on detected faults; 66.3 % on all true faults; 51.5 % excluding Jamming | [`track_a_metrics.json`](data/processed/track_a_metrics.json) |
| Cascade | Congestion-family accuracy on all true faults | | 75.0 % | [`validation.json`](data/processed/validation.json) |
| Retrieval | Top-1 cosine similarity on 382 flagged test windows | | 0.59 to 0.75 (median 0.72) | [`track_a_metrics.json`](data/processed/track_a_metrics.json) |
| Track C (citations) | Correct 3GPP specification, C1 / C2 / C3 (n = 30) | | 5 / 5 / **21** (16.7 % / 16.7 % / **70.0 %**) | [`track_c_summary.json`](data/processed/track_c_v3/track_c_summary.json) |
| Track C (citations) | Correct when the fault type is right (n = 19) | | 4 / 4 / **19** | [`track_c_summary.json`](data/processed/track_c_v3/track_c_summary.json) |
| Track B (G-Eval) | Citation-weighted composite, 24 explanations | | **3.75 / 5** (SD 0.28); 3.79 / 5 equal-weight | [`track_b_geval_results.json`](data/processed/track_b_geval_results.json) |
| Track D (TeleQnA) | Accuracy, 600 questions: no retrieval / ungated RAG / gated at 0.50 | | 76.33 % / 51.83 % / 76.00 % | [`track_d_full.json`](data/processed/track_d_full.json) |

All 9 remaining Track C misses start upstream of the explanation layer (6 windows the detector
missed, 3 faults classified as another type). When the fault type is right, the cited standard was
correct in 19 of 19 windows.

## Architecture

```
TelecomTS window (18 channels x 128 steps, 100 ms sampling)
        |
        v
Layers 1-2 (offline): CatBoost cascade + TreeSHAP
        Model 1: fault / normal (64 window statistics, threshold 0.5)
        Model 2: fault type, 11 classes (496 features, trained on 935 real + 4,250 synthetic windows)
        -> data/processed/layer2_output_sessionsplit_phase2.json
        |
        v
Retrieval: past incidents of the PREDICTED fault type (ChromaDB metadata filter),
        ranked by KPI-evidence similarity; the query carries no label
        |
        v
Prompt: SHAP evidence + retrieved incidents + the fault's alignment-table row (v1.2)
        + per-window RCA evidence (data/processed/rca_evidence.json)
        |
        v
LLM (Gemini; Groq or Ollama as alternatives) -> structured JSON explanation
        |
        v
Citation validation: TS/TR format check + membership in the fault's alignment row
        |
        v
Streamlit dashboard (6 panels)
```

The alignment table is the only source of standards in the prompt. Retrieval supplies concrete KPI
patterns from comparable incidents; it does not decide the fault type, because the dataset's
troubleshooting tickets follow one text template per fault class (see the final report, Section 5.6).

### Alignment table (v1.2)

[`data/alignment_table.json`](data/alignment_table.json) maps each of the 11 fault types to a 3GPP
reference and an O-RAN component. Two rows are directly supported, eight are mechanism-level, and
Jamming is ticket-retrieval-only with no 3GPP mapping. Earlier versions are kept in
[`configs/alignment table/`](configs/alignment%20table/README.md).

| # | Fault type | Primary 3GPP reference | O-RAN component | Support class |
|---|---|---|---|---|
| 1 | Antenna Failure | TS 38.141-1 §4.2.1, §7.2 | O-RU | Mechanism-level |
| 2 | Buffer Overflow (Gradual Buildup) | TS 38.321 §6.1.3.1, §5.4.5 | O-DU | Mechanism-level |
| 3 | Co-Channel Interference (Mild) | TS 38.141-1 §7.8 | O-RU | Mechanism-level |
| 4 | Co-Channel Interference (Severe) | TS 38.141-1 §7.8; TS 38.104 §4.9 | O-RU | Mechanism-level |
| 5 | Doppler Shift (Severe) | TR 38.901 §7.6.6, §7.6.10 | O-DU / O-RU | Mechanism-level |
| 6 | Faulty Handover Algorithm (Too Frequent) | TS 38.331 §5.5.4.4, §5.5.4.1 | Near-RT RIC | Mechanism-level |
| 7 | Faulty RF Filters (Temporal) | TS 38.141-1 §7.5, §7.6; TS 38.104 §6.6 | O-RU | Mechanism-level |
| 8 | High Network Congestion (Gradual Buildup) | TS 28.552 §5.1.1.2.1, §5.1.1.2.2 | O-DU | Directly supportable |
| 9 | High Network Congestion (Sudden Spike) | TS 28.552 §5.1.1.2.1, §5.1.1.2.2 | O-DU | Directly supportable |
| 10 | Resource Allocation Bugs | TS 38.321 §5.4.3.1.3 | O-DU / Near-RT RIC | Mechanism-level |
| 11 | Jamming | none | none | Ticket-retrieval-only |

## Getting started

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A [Gemini API key](https://aistudio.google.com/apikey) (the free tier works); a
  [Groq](https://console.groq.com/) key or a local [Ollama](https://ollama.com/) server are alternatives

### Install

```bash
git clone https://github.com/KPI-RAG/kpi-rag.git
cd kpi-rag
uv sync                      # installs locked dependencies, CPU-only torch
cp .env.example .env         # then set GEMINI_API_KEY in .env
```

The trained models ([`data/models/`](data/models/)), the Layer 2 output, the RCA evidence and the
ChromaDB index ([`data/chroma_db/`](data/chroma_db/)) are committed, so the dashboard and the
evaluation scripts run without the raw dataset.

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
[`build_index.py`](scripts/build_index.py) checks every ticket against the ground truth before
indexing. The index holds the 935 anomalous training windows only.

## Usage

### Dashboard

> [!TIP]
> **Live Demo:** Try the deployed dashboard at [**kpi-rag-group15.streamlit.app**](https://kpi-rag-group15.streamlit.app/).

To run the dashboard locally:

```bash
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

1. Choose an input in the sidebar: **Use example**, **Browse Dataset Windows** (the 382 test
   windows flagged as faults, filterable by fault type) or **Upload JSON** (a `ClassifierOutput`
   payload).
2. Click **Generate Explanation**.
3. Read the diagnosis, the citation check, the retrieved incidents and the RCA evidence.

The six panels live in [`dashboard/components/`](dashboard/components/): detection status,
SHAP top contributors, KPI window statistics (mean ± std), explanation (root cause, 3GPP
reference with a validity badge, O-RAN component, action, reasoning), similar historical
incidents, and RCA evidence.

To deploy on Streamlit Community Cloud, set the main file to `dashboard/app.py`, use Python 3.11
and add `GEMINI_API_KEY` under *Secrets*.

### Command line

```bash
uv run python scripts/run_pipeline.py --input path/to/classifier_output.json --output explanation.json
```

## Evaluation

| Track | Measures | Command | Output |
|---|---|---|---|
| Layers 1–2 | Models, features, synthetic data | see [`Data_and_Models/`](Data_and_Models/README.md) | [`validation.json`](data/processed/validation.json) |
| A | Detection, classification, retrieval | `uv run python scripts/run_eval_track_a.py` | [`track_a_metrics.json`](data/processed/track_a_metrics.json) |
| A | Retrieval design comparison (needs raw data) | `uv run python scripts/experiment_retrieval_design.py` | printed |
| B | Explanation quality (G-Eval LLM judge) | see below; `--scores <jsonl>` re-aggregates pre-scored records | [`track_b_geval_results.json`](data/processed/track_b_geval_results.json) |
| C | Citation ablation, 3 prompt conditions | see below | [`data/processed/track_c_v3/`](data/processed/track_c_v3/) |
| D | Knowledge scope on TeleQnA | see below | [`track_d_full.json`](data/processed/track_d_full.json) |

### Layers 1–2: models and experiments

[`Data_and_Models/`](Data_and_Models/README.md) holds the notebooks (`1_First_prototype/`,
`2_Model_improvement/`) and the numbered experiment scripts and results (`3_Final_system/`, steps
01 to 12c). Results use a session-level split (33 recording sessions: 28 train, 5 test), because a
random window split leaked near-copies into the test set (31.8 % of test windows had a
near-identical training window). The two fixes that cut false alarms from 1,105 to 116 were
correcting a double class-imbalance correction and a parsing bug that read `1.e-05` as two numbers.
See [`Data_and_Models/3_Final_system/README.md`](Data_and_Models/3_Final_system/README.md).

### Track B: explanation quality

An LLM judge (`qwen/qwen3.8-27b`, temperature 0.0, default) scores the Track C condition-C3
explanations on four dimensions, 1 to 5. Of the 30 sampled windows, 24 have an explanation; the
other 6 were missed by the detector and are excluded, not scored as zero. No judge errors occurred.

| Dimension | Mean | SD | Min | Max |
|---|---|---|---|---|
| Fault specificity | 4.04 | 0.36 | 3 | 5 |
| Actionability | 3.79 | 0.51 | 3 | 5 |
| Causal soundness | 3.75 | 0.61 | 2 | 5 |
| Citation validity | 3.58 | 0.58 | 3 | 5 |
| Composite (citation 40 %, others 20 %) | 3.75 | 0.28 | | |

Citation validity scored 4 or above in 13 of 24 explanations (54.2 %). Track C checks only the
specification number, while the judge also asks for the clause, so the two rates differ.

```bash
export GROQ_API_KEY=...      # or pass --api-key and --base-url
uv run python scripts/run_eval_track_b.py \
  --explanations data/processed/track_c_v3/track_c_explanations.jsonl \
  --alignment data/alignment_table.json \
  --output data/processed/track_b_geval_results.json
```

### Track C: citation ablation

Three test windows per true fault type (10 types; Jamming has no applicable 3GPP clause), seed 42,
each run through the full system. The conditions share one instruction block and differ only in
context: C1 gets the label and SHAP evidence, C2 adds three past incidents, C3 adds the alignment
row and the window's RCA evidence. A citation counts as correct only against the **true** fault's
alignment row; detector misses count as failures.

| Answer rule | Correct (n = 30) | Fault type right (n = 19) | Valid vs predicted row (n = 24) |
|---|---|---|---|
| C1 LLM: label + SHAP | 5/30 | 4/19 | 7/24 |
| C2 LLM: + past incidents | 5/30 | 4/19 | 7/24 |
| C3 LLM: + alignment row + RCA evidence | **21/30** | **19/19** | 24/24 |
| No LLM: always cite TS 38.141-1 | 12/30 | | |
| No LLM: copy the predicted fault's row | 21/30 | | |

C3 equals the copy-row rule by design: the LLM adopts the standard the table provides, and the
validator holds it to that row. With n = 30 from a few recording sessions, treat the counts as
descriptive.

```bash
uv run python scripts/run_eval_track_c.py --summarise-only                    # re-score the stored run, no LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/my_run      # new run, 72 LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/my_run --dry-run   # 1 window per fault
```

A run never overwrites existing results unless `--force` is given.

### Track D: TeleQnA knowledge scope

600 TeleQnA questions (200 each from Lexicon, Standards overview and Standards specifications) are
answered by `gemini-3.5-flash-lite` with and without chunks retrieved from the 935-incident index.
The index describes KPI anomalies, not standards text, so only 13 of 600 questions (2.17 %) reach
the coverage threshold of 0.50.

| Category | Supported (score ≥ 0.50) | No retrieval | Ungated RAG |
|---|---|---|---|
| Lexicon | 2 | 87.0 % | 41.5 % |
| Standards overview | 1 | 75.0 % | 53.0 % |
| Standards specifications | 10 | 67.0 % | 61.0 % |
| **All (600)** | **13** | **76.33 %** (458) | **51.83 %** (311) |

Injecting irrelevant context into every prompt cost 24.5 percentage points. A gate that falls back
to the model alone when the top retrieval score is below 0.50 gives 76.00 % (456 of 600). On the 13
supported questions, the baseline scored 12 and ungated RAG 10.

```bash
uv run python scripts/run_eval_track_d.py --config configs/config.yaml --audit-only   # coverage only
uv run python scripts/run_eval_track_d.py --config configs/config.yaml --output data/processed/track_d_full.json
```

## Testing

```bash
uv run pytest tests/ -q
```

The suite has 29 test modules (21 unit in [`tests/`](tests/), 8 integration in
[`tests/integration/`](tests/integration/)) and 210 tests: 207 pass and 3 are skipped without `GROQ_API_KEY`.
The test session restores `data/chroma_db/` afterwards, because ChromaDB rewrites its files
whenever it is opened.

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
| `llm.gemini_model` | `gemini-3.5-flash-lite` | Fallbacks: `gemini-2.5-flash-lite`, `gemini-2.0-flash-lite` |
| `llm.temperature` | `0.1` | |
| `llm.max_retries` | `3` | Backs off on rate limits; template fallback at the end |
| `llm.min_request_interval_s` | `2.0` | Minimum gap between LLM calls |
| `data.n_features` | `496` | Model 2 feature count |
| `evaluation.citation_validity_threshold` | `0.70` | Pre-registered pass mark |

## Repository layout

```
.
├── configs/
│   ├── config.yaml              # retrieval, LLM, data and evaluation settings
│   └── alignment table/         # alignment table versions v1.0, v1.1, v1.2 and their README
├── Data_and_Models/             # Layers 1-2: notebooks, experiment scripts, results, final models
├── src/                         # library code
│   ├── schema.py                #   Pydantic models (11 fault types)
│   ├── data_loader.py           #   TelecomTS loader and train/test split
│   ├── kg_indexer.py            #   ChromaDB indexing
│   ├── rag_query.py             #   retrieval with fault-type filter
│   ├── rca_loader.py            #   per-window RCA evidence for the prompt
│   ├── build_rca_evidence.py    #   builds the RCA evidence file
│   ├── generate_alignment_table.py
│   ├── rca_utils.py             #   KPI evidence assessment and RCA utilities
│   ├── llm_explainer.py         #   prompts, LLM backends, retries, citation validation
│   ├── evaluator.py             #   Track B and C metrics
│   └── ml_model.py, utils.py, config_loader.py, COLORS.py
├── scripts/                     # command-line entry points (see scripts/README.md)
│   └── Standards_inspection/    #   scripts that search the 3GPP knowledge graph
├── dashboard/                   # Streamlit app and its 6 panels
├── tests/                       # 21 unit and 8 integration test modules
├── data/
│   ├── alignment_table.json     # fault -> 3GPP/O-RAN mapping, v1.2
│   ├── models/                  # CatBoost Model 1 and Model 2
│   ├── indices/                 # session-split train/test indices
│   ├── chroma_db/               # retrieval index (committed for Streamlit Cloud)
│   └── processed/               # Layer 2 output, RCA evidence, Track A to D results
├── pyproject.toml, uv.lock      # environment (uv)
└── requirements.txt             # pinned export of uv.lock
```

More detail: [`Data_and_Models/README.md`](Data_and_Models/README.md),
[`configs/alignment table/README.md`](configs/alignment%20table/README.md),
[`scripts/README.md`](scripts/README.md),
[`scripts/Standards_inspection/README.md`](scripts/Standards_inspection/README.md).

## Known limitations

- Only 10 of the 28 training sessions contain faults, so small gains cannot be proven; the
  held-out set has 5 sessions and Track C has n = 30. Treat the counts as descriptive.
- High Network Congestion (Gradual) and Buffer Overflow are often confused; 30 of 37 held-out
  windows of the first are named as the second.
- Track B and Track C measure citations differently (clause depth versus specification number),
  so their rates are not comparable.

## Team

| Member | Main contributions |
|---|---|
| Ahmed Mohamed Al Shobaki (team leader) | Retrieval and its redesign, prompt design, LLM explanation layer and citation validation, Track C, dashboard, deployment, test suite |
| Farida El Ghabary | Synthetic training data and the Model 2 recipe |
| Raneem Alnaghy | Recording-session reconstruction and the session-level split; false-alarm filter study |
| Rodina Khallaf | 3GPP Release 19 / O-RAN alignment table; Track B (G-Eval) and Track D (TeleQnA) evaluations |

## Acknowledgements

Prof. Yuanzhu Chen (supervisor); Capgemini (industry partner); Queen's University and its Frontenac
cluster. The TelecomTS dataset is by Feng et al.,
"TelecomTS: A multi-modal observability dataset for time series and language analysis",
[arXiv:2510.06063](https://arxiv.org/abs/2510.06063), 2025. Track D uses the
[TeleQnA](https://huggingface.co/datasets/netop/TeleQnA) benchmark
([arXiv:2310.15051](https://arxiv.org/abs/2310.15051)).

## License

Distributed under the [MIT License](LICENSE). The TelecomTS and TeleQnA datasets are subject to their
own licenses on Hugging Face.
