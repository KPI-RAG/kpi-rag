# KPI-RAG: 5G Network Fault Diagnosis with Retrieval-Augmented Generation

An explainability framework for 5G network fault classification that combines SHAP-based feature attribution, knowledge-graph retrieval, and LLM-generated explanations grounded in 3GPP standards.

## Architecture

```
Classifier → SHAP top-3 KPIs
                 ↓
           ChromaDB retrieval (similar tickets)
                 ↓
           Alignment table (3GPP TS → fault mapping)
                 ↓
           LLM (Groq / Ollama) → structured explanation
                 ↓
           Citation validation (TS XX.XXX format + table lookup)
                 ↓
           Streamlit dashboard (5 panels)
```

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager
- [Groq API key](https://console.groq.com/) (free tier works)

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
# Edit .env and add your GROQ_API_KEY

# Or set directly:
$env:GROQ_API_KEY="gsk_..."
$env:PYTHONPATH="."
```

### 3. Place raw data

Download the TelecomTS dataset and place JSONL files under:

```
data/raw/          # 33 JSONL files (nested structure supported)
```

### 4. Build the ChromaDB index

```bash
uv run python scripts/build_index.py
```

This indexes ~1,235 anomaly tickets into ChromaDB at `data/chroma_db/`.

### 5. Run the pipeline (single explanation)

```bash
uv run python scripts/run_pipeline.py
```

### 6. Run the dashboard

```bash
$env:PYTHONPATH="."
$env:GROQ_API_KEY="gsk_..."
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

## Evaluation

### Track B — Full-system evaluation

```bash
uv run python scripts/run_eval_track_b.py --output data/processed/ --n-samples 30
```

### Track C — Ablation study (3 conditions)

| Condition | Context provided |
|-----------|-----------------|
| C1 | Label + SHAP only |
| C2 | Label + SHAP + retrieved tickets |
| C3 | Full system (tickets + alignment table) |

```bash
# Dry run (1 sample per fault = 30 LLM calls)
uv run python scripts/run_eval_track_c.py --output data/processed/ --dry-run

# Full run (3 samples per fault = 90 LLM calls)
uv run python scripts/run_eval_track_c.py --output data/processed/
```

Output files:
- `data/processed/track_c_explanations.jsonl` — generated explanations
- `data/processed/track_c_scores_template.jsonl` — human annotation template

## Testing

```bash
uv run pytest tests/ -v
```

16 test modules covering schema validation, RAG query, LLM explainer, evaluator, Track B/C scripts, and all 5 dashboard panels.

## Project Structure

```
project/
├── configs/
│   ├── config.yaml              # RAG, LLM, data, eval settings
│   └── alignment_table.json     # 10-row fault→3GPP mapping (DRAFT)
├── src/
│   ├── schema.py                # Pydantic models (11 fault types)
│   ├── config_loader.py         # YAML config loader
│   ├── data_loader.py           # TelecomTS JSONL loader
│   ├── kg_indexer.py            # ChromaDB indexing
│   ├── rag_query.py             # Cosine similarity retrieval
│   ├── llm_explainer.py         # Groq/Ollama LLM + 3-condition prompts
│   ├── evaluator.py             # G-Eval scoring + Track B/C metrics
│   └── utils.py                 # Logging, 3GPP ref validation
├── scripts/
│   ├── build_index.py           # Index builder CLI
│   ├── run_pipeline.py          # Single-explanation CLI
│   ├── run_eval_track_b.py      # Track B evaluation CLI
│   └── run_eval_track_c.py      # Track C ablation CLI
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   └── components/              # 5 dashboard panels
├── tests/                       # 16 test modules
├── data/
│   ├── raw/                     # TelecomTS JSONL (gitignored)
│   ├── chroma_db/               # ChromaDB store (gitignored)
│   └── processed/               # Evaluation outputs
├── configs/config.yaml
├── pyproject.toml
├── .env.example
└── context.md                   # Development state tracking
```

## Configuration

Key settings in `configs/config.yaml`:

| Setting | Value | Notes |
|---------|-------|-------|
| `rag.embedding_model` | `all-MiniLM-L6-v2` | Sentence-BERT |
| `rag.cosine_threshold` | `0.35` | Provisional — calibrate Week 9 |
| `rag.top_k` | `5` | Retrieved tickets per query |
| `llm.backend` | `groq` | Also supports `ollama` |
| `llm.groq_model` | `llama-3.1-8b-instant` | Free tier compatible |
| `llm.max_retries` | `2` | Template fallback after failures |

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
