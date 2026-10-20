# Source Code

This directory contains the core implementation of KPI-RAG, the end-to-end 5G network fault diagnosis and explanation pipeline. The code is organized around the project’s main workflow:

1. Load telecom telemetry and validate the raw dataset format.
2. Run the CatBoost fault-detection and fault-classification models.
3. Build or query a ChromaDB retrieval index of historical incidents.
4. Construct RCA evidence from model output, KPI statistics, and SHAP attribution.
5. Ground explanations in a TelecomTS / 3GPP / O-RAN alignment table.
6. Prompt an LLM to generate a structured root-cause explanation with citation validation.

The project is designed for research-grade explainability: the model identifies likely faults, retrieval finds similar historical cases, and standards grounding provides the factual basis for the final explanation.

## Why this module exists

The `src/` package is not a generic utility library. It is the operational engine of KPI-RAG and is tightly coupled to:

- the TelecomTS dataset structure,
- the CatBoost pipeline used for fault detection and fault-type classification,
- the ChromaDB retrieval layer,
- the alignment-table standards grounding workflow,
- and the LLM-based explanation pipeline.

The implementation aims to keep the pipeline transparent: the model explains what mattered, retrieval provides comparable cases, and the alignment table supplies the telecom-domain grounding.

## Module map

- `schema.py` — Pydantic models for fault classes, classifier outputs, and retrieval results.
- `config_loader.py` — YAML/env configuration loading for the project’s runtime settings.
- `data_loader.py` — loads TelecomTS JSONL records, filters anomalous tickets, and enforces train/test splits.
- `kg_indexer.py` — creates and searches the vector index used for similar-incident retrieval.
- `rag_query.py` — builds text queries from classifier outputs and retrieves past incidents by KPI similarity.
- `rca_loader.py` — loads per-window RCA evidence used to support explanation generation.
- `llm_explainer.py` — prompts, model backends, citation validation, and explanation synthesis logic.
- `evaluator.py` — computes evaluation metrics for Track B and Track C experiments.
- `ml_model.py` — loads and applies the CatBoost model pipeline.
- `build_rca_evidence.py` — assembles structured RCA evidence from Layer 2 predictions and alignment-table rows.
- `generate_alignment_table.py` — builds and validates the fault-to-3GPP/O-RAN mapping table.
- `utils.py` — shared helper functions used across the package.
- `COLORS.py` — terminal color formatting used in console output.
- `__init__.py` — package marker for the `src` module.

## End-to-end data flow

```text
TelecomTS window
    |
    v
Model 1: normal vs. fault
    |
    v
Model 2: fault-type classification
    |
    v
SHAP-based KPI attribution
    |
    v
Similar-incident retrieval (ChromaDB)
    |
    v
Alignment-table grounding (3GPP / O-RAN)
    |
    v
LLM explanation + static citation validation