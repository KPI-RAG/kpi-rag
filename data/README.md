# RAG Data

This directory contains the core data artifacts used by the KPI-RAG pipeline for standards-grounded retrieval, root cause analysis, and performance evaluation.

It includes the retrieval database, dataset index splits, serialized model artifacts, and processed outputs generated during experimentation.

## Repository layout

```text
RAG_data/
├── alignment_table.json
├── chroma_db/
│   ├── chroma.sqlite3
│   └── 66be7e31-95ff-4668-aa08-3a4655ac4731/
├── indices/
│   ├── train_idx_sessionsplit.npy
│   └── test_idx_sessionsplit.npy
├── models/
│   ├── model1.cbm
│   └── model2.cbm
└── processed/
    ├── feature_spec.json
    ├── layer2_output_sessionsplit_phase2.json
    ├── layer2_rag_handoff_sessionsplit.json
    ├── log.txt
    ├── normal_means.json
    ├── rca_evidence.json
    ├── track_a_metrics.json
    ├── track_b_geval_results.json
    ├── track_c_v3/
    ├── track_d_full.json
    └── validation.json
```

## Contents

| Path | Description |
| --- | --- |
| `alignment_table.json` | Alignment metadata used to map standards, features, and RCA relationships. |
| `chroma_db/` | Local Chroma vector store used for retrieval-based semantic search. |
| `indices/` | Train/test dataset index split files used for evaluation workflows. |
| `models/` | Model artifacts stored in `.cbm` format. |
| `processed/` | Intermediate and final outputs created during analysis and evaluation. |

## Key artifacts

### `alignment_table.json`
Stores structured alignment data linking the retrieval corpus with relevant standards and analysis targets.

### `chroma_db/`
Contains the vector database used by the retrieval layer to search and retrieve candidate evidence relevant to a query or incident.

### `indices/`
Provides dataset split identifiers used to train and evaluate the model on session-based partitions:
- `train_idx_sessionsplit.npy`
- `test_idx_sessionsplit.npy`

### `models/`
Holds binary model artifacts used in downstream analysis or inference pipelines.

### `processed/`
Contains generated data products such as:
- feature specifications
- layer-2 RAG and RCA outputs
- evaluation metrics
- evidence traces
- validation and logging summaries

## Notable processed files

- `feature_spec.json` — feature schema used in analysis.
- `layer2_output_sessionsplit_phase2.json` — phase-2 output from the model pipeline.
- `layer2_rag_handoff_sessionsplit.json` — handoff data between retrieval and downstream reasoning.
- `rca_evidence.json` — evidence used to explain root cause conclusions.
- `track_a_metrics.json` — metrics for track A evaluation.
- `track_b_geval_results.json` — G-Eval results for track B.
- `track_d_full.json` — full outputs for track D.
- `validation.json` — validation summaries and metadata.

## Notes

This directory is intended to hold generated research and pipeline artifacts rather than source code. It is essential for:
- reproducing retrieval experiments,
- validating results,
- inspecting explainable evidence,
- and supporting downstream root-cause analysis workflows.

If you are working with the repository locally, keep this directory synchronized with the corresponding model and preprocessing code to preserve reproducibility.