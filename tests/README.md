# Tests

This directory contains the automated validation suite for KPI-RAG. The tests cover the project’s core behavior: configuration loading, dataset processing, retrieval quality, model pipeline execution, explanation generation, citation validation, and dashboard behavior.

The test suite is designed to protect the integrity of the full 5G fault-diagnosis pipeline, especially where the system combines machine learning, retrieval, and standards-grounded explanation.

## Test philosophy

KPI-RAG is a research and engineering pipeline with multiple moving parts:
- raw TelecomTS data ingestion,
- fault detection/classification,
- ChromaDB retrieval,
- LLM-based explanation generation,
- citation validation against an alignment table,
- and end-to-end pipeline execution.

Because of that, the tests do not only validate isolated functions. They also check that the system behaves correctly when these components work together.

## Directory structure

The test suite is organized around the project’s functional areas:

- `conftest.py` — shared pytest session setup, including restoring the committed ChromaDB files after tests run.
- `test_config_loader.py` — validates configuration reading and application settings.
- `test_data_loader.py` — verifies raw telecom data loading and train/test split logic.
- `test_build_index.py` — tests vector index creation and index maintenance.
- `test_kg_indexer.py` — validates collection creation, metadata handling, and retrieval behavior.
- `test_rag_query.py` — checks query construction and fault-type-specific retrieval logic.
- `test_llm_explainer.py` — validates prompt construction, LLM responses, and citation-checking logic.
- `test_evaluator.py` — covers evaluation logic for explanation-quality and citation metrics.
- `test_run_pipeline.py` — validates the end-to-end inference flow.
- `test_run_eval_track_a.py` — checks Track A metrics (detection/classification/retrieval).
- `test_run_eval_track_b.py` — verifies Track B explanation scoring.
- `test_run_eval_track_c.py` — tests citation-ablation evaluation logic.
- `test_schema.py` — validates Pydantic models and domain objects.
- `test_phase2_data.py` — tests Phase 2 data processing assumptions and outputs.
- `test_phase2_models.py` — validates model-related assumptions in the training pipeline.
- `test_phase2_pipeline.py` — exercises the Phase 2 end-to-end processing flow.
- `test_review3_fixes.py` — regression tests for earlier fixes and review feedback.
- Dashboard-related tests:
  - `test_detection_panel.py`
  - `test_explanation_panel.py`
  - `test_kpi_signal_panel.py`
  - `test_shap_panel.py`
  - `test_sources_panel.py`

## Important test setup note

`conftest.py` contains an autouse session fixture that restores the committed `data/chroma_db/` directory after tests finish.

This is necessary because:
- ChromaDB writes files to disk when a persistent client is opened,
- and those writes can modify repository-managed state even during read-only usage.

The cleanup pattern ensures that the repo remains consistent after testing.

## Coverage areas

The tests cover:

- configuration integrity
- data loading and validation
- deterministic split behavior
- retrieval correctness
- query-generation quality
- LLM explanation generation and structure
- standards-citation validation
- end-to-end pipeline execution
- regression prevention for key fixes
- dashboard output and panel behavior

## Test execution

Run the full suite from the repository root:

```bash
uv run pytest tests/ -q