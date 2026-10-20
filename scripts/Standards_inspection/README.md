# Standards inspection

This folder contains the scripts used to inspect the 3GPP / O-RAN knowledge graph and identify which standards entities are relevant to the 5G fault types in KPI-RAG. The workflow turns raw graph data into filtered evidence for fault explanations and citations.

The work is centered on the files under `data/3GPP_KG/`, especially:

- `data/3GPP_KG/tkg/rel19_3gpp_telecom_kg.graphml`
- `data/3GPP_KG/gsma_entities.csv`
- `data/3GPP_KG/fault_coverage_candidates.csv`
- `data/3GPP_KG/fault_graph_evidence.csv`

## Purpose

The goal is to answer a simple but critical question: for each telecom fault class, which entities and relationships in the standards graph best support the explanation and citation layer?

This folder performs that search in stages:

1. Extract entities from the GraphML graph.
2. Search text fields for fault-related terminology.
3. Rank the strongest candidate standards using fault-specific keywords.
4. Inspect neighboring graph evidence to understand the local context.
5. Validate the final evidence and keep only the most relevant items.

## Script overview

### `extract_gsma_entities.py`
Extracts nodes from the GraphML knowledge graph and writes a CSV with fields such as `entity_id`, `entity_type`, `description`, `source_id`, `file_path`, and `release`.

This is usually the first step because the later scripts operate on the flattened entity table rather than the raw XML graph.

### `search_fault_coverage.py`
Performs a broad regex search across the extracted GSMA/3GPP entities for generic telecom fault terms such as:

- antenna / antenna failure
- interference
- congestion
- resource allocation
- handover
- buffer overflow
- RF filter
- doppler

It produces `fault_coverage_candidates.csv` with the matched fault labels and the terms that triggered each match.

### `rank_fault_candidates.py`
Takes the broad candidate list and assigns a more informative score based on fault-specific terminology. It keeps the strongest candidates per fault class and writes `fault_top_candidates.csv` for further review.

### `targeted_fault_search.py`
Performs a deeper graph-based search on the GraphML network using fault-specific search dictionaries (for example Doppler Shift, Jamming, Antenna Failure, Handover Failure, and Resource Allocation issues).

### `semantic_fault_search.py`
A more semantic-oriented variation of the graph search. It searches for technical phrases and related terminology around each fault category and returns a `semantic_fault_search.csv` output containing the best matches.

### `doppler_broad_search.py`
A focused Doppler-related search intended to capture variants such as:

- doppler shift
- frequency offset
- time-varying channel
- fading
- channel variation due to mobility

This script produces a specific candidate list for Doppler-related standards evidence.

### `validate_fault_graph.py`
Matches the candidate entities back to graph nodes and extracts their surrounding relationships to build a richer evidence table. This step produces `fault_graph_evidence.csv`, which contains both entity-level and relationship-level evidence.

### `inspect_fault_evidence.py`
Ranks the graph evidence by relevance and outputs a review file like `fault_evidence_review.csv`. It uses a relevance score based on:

- explicit fault terminology
- edge description metadata
- keywords and conditions attached to the graph relationships
- entity descriptions and neighboring concept text

This is the final evidence audit step before the standards result is trusted by the explanation pipeline.

### `inspect_gsma_kg.py`
Checks the underlying text chunk dataset and summarizes basic metadata such as:

- total chunk count
- how many chunks are non-empty
- series distribution
- unique TS identifiers

This helps sanity-check the raw standards corpus before graph searches begin.

## Typical workflow

A practical run order is:

```bash
python scripts/Standards_inspection/extract_gsma_entities.py
python scripts/Standards_inspection/search_fault_coverage.py
python scripts/Standards_inspection/rank_fault_candidates.py
python scripts/Standards_inspection/targeted_fault_search.py
python scripts/Standards_inspection/semantic_fault_search.py
python scripts/Standards_inspection/validate_fault_graph.py
python scripts/Standards_inspection/inspect_fault_evidence.py
```

## Inputs and limitations

- The `data/3GPP_KG/` files named above are **not committed** to this repository: the GraphML knowledge graph must be obtained separately and placed there before running these scripts.
- The two full standards knowledge graphs are too large to upload to this repository and are therefore hosted externally on Hugging Face:
  - Telecom-KG-Rel19: https://huggingface.co/datasets/GSMA/telecom-kg-rel19
  - O-RAN Specification Knowledge Graph: https://huggingface.co/datasets/GSMA/oran_spec_knowledge_graph
- The scripts hard-code Windows-style paths (for example `data\\3GPP_KG\\gsma_entities.csv`), so they run as written on Windows only. Edit the path constants at the top of each script to run them on other platforms.
- The outputs of this workflow are already summarised in `data/alignment_table.json`; the project does not need these scripts at runtime.
