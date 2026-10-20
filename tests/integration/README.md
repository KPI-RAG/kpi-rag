# Integration Tests

This directory contains the repository's integration-style validation suite for KPI-RAG. These tests exercise the system as a connected pipeline rather than checking isolated utility functions in isolation.

The focus is on the standards-grounded 5G fault-diagnosis workflow: schema validation, query construction, retrieval, explanation generation, and end-to-end pipeline behavior across the supported fault types.

## Purpose

The tests in this folder verify that KPI-RAG behaves correctly when multiple subsystems interact:

- anomaly schema and classifier output validation
- evidence-only RAG query generation
- ChromaDB retrieval and document search behavior
- alignment-table validation for standards grounding
- LLM-based explanation generation
- evaluator checks for explanation quality and citation validity
- end-to-end execution of the diagnosis pipeline

## Coverage summary

The directory contains the following test modules:

- `conftest.py` — shared pytest fixtures and reusable payloads for all supported fault classes
- `test_01_schema_all_faults.py` — validates all 11 fault categories and ensures SHAP sign and direction metadata remain consistent
- `test_02_rag_query_all_faults.py` — verifies that generated retrieval queries are evidence-based, non-empty, and do not leak the fault label into the prompt text
- `test_03_chromadb_retrieval.py` — checks retrieval behavior against the persistence-backed vector store
- `test_04_alignment_table.py` — validates the standards-alignment lookup used to ground explanations
- `test_05_llm_explainer_all_faults.py` — exercises the LLM explainer across all supported fault types
- `test_06_evaluator_all_faults.py` — verifies evaluation logic for explanation and citation quality
- `test_07_full_pipeline.py` — runs broader pipeline-level regression checks
- `test_08_full_rag_cycle_real.py` — validates end-to-end RAG flow using realistic data and execution paths

## Shared fixtures

`conftest.py` defines a session-scoped `all_payloads` fixture containing one realistic `ClassifierOutput` payload for each supported anomaly type. This allows the same validation checks to be applied across the entire fault catalog without duplicating setup logic.

## Running the tests

From the repository root:

```bash
pytest tests/integration -q
```

Or:

```bash
python -m pytest tests/integration -q
```

## Notes

These tests are intended to protect the integrity of KPI-RAG's end-to-end diagnosis pipeline, especially where retrieval, model reasoning, and standards-grounded explanation must remain coherent across many fault scenarios.