# Alignment table

This folder stores the versioned alignment tables used to ground each TelecomTS fault in the relevant 3GPP / O-RAN standard and to keep the citation layer auditable.

The files in this directory document the evolution of the standard-to-fault mapping used by KPI-RAG and preserve a traceable record of the changes made during validation.

## Folder contents

This directory contains the following JSON files:

- `alignment_table_v1.0.json` — first draft of the alignment table
- `alignment_table_v1.1.json` — updated version after the first validation pass
- `alignment_table_v1.2.json` — final reviewed version used by the project

## Purpose

Each JSON file defines a fault-to-standard alignment table for the 10 synthetic fault types used in TelecomTS and the special-case Jamming row. Every row maps a fault label to:

- the governing 3GPP / TR clause(s)
- the relevant O-RAN component
- the causal mechanism
- expected KPI channels
- the support class and evidence strength
- the validation status
- notes describing what was confirmed, corrected, or deferred

## Why multiple versions?

The project went through multiple rounds of standards review. The version history preserves earlier decisions instead of overwriting them.

### v1.0
- Initial alignment table draft
- Includes all 10 synthetic fault rows and the Jamming exclusion note
- Some mappings were still broad or incomplete and required validation follow-up
- Notable entries requiring review included Doppler and resource allocation mappings

### v1.1
- First revision after validation review
- Corrected the Doppler section using ETSI TR 38.901 V19.1.0
- Corrected the resource allocation row by moving from a broad TS 38.321 section to the specific §5.4.3.1.3 subclause
- Kept a historical record of row updates and causal text refinements

### v1.2
- Final project version
- Completes the remaining validation work and clears all pending validation items
- Synchronizes causal mechanisms with the actual onset patterns of each anomaly type
- Finalizes the Jamming mapping as a ticket-retrieval-only case with no direct 3GPP clause
- This is the version intended for the main KPI-RAG citation and explanation pipeline

## Schema conventions

Each row in the alignment table follows the same pattern:

- `row_id`
- `telecomts_fault`
- `domain`
- `onset_pattern`
- `expected_kpis`
- `affected_kpis`
- `3gpp_reference`
- `3gpp_release`
- `3gpp_version`
- `clause_text`
- `causal_mechanism`
- `causal_mechanism_sources`
- `oran_component`
- `oran_component_rationale`
- `oran_spec_reference`
- `relation_type`
- `support_class`
- `evidence_strength`
- `kg_evidence_summary`
- `validation_status`
- `notes`

## Important interpretation rules

- `support_class` distinguishes between direct standards support and mechanism-level support.
- `relation_type` is one of:
  - `governed-by`
  - `specified-in`
  - `measured-by`
  - `not-applicable`
- `Jamming` is intentionally excluded from direct 3GPP standard mapping because it is a real over-the-air RF attack rather than a standards-defined network behavior.
- `affected_kpis` are refreshed from the trained model's SHAP-ranked channels; these values are separate from the standards evidence and do not change the governing 3GPP/O-RAN mapping.

## Recommended usage

- Use `alignment_table_v1.2.json` as the current production alignment table.
- Use `alignment_table_v1.0.json` and `alignment_table_v1.1.json` only for historical review, audit tracing, or publication notes.
- If the project later updates the alignment logic, add a new versioned file rather than overwriting the prior one.

## Validation note

The alignment table is not a static truth source; it is a reviewed mapping used to constrain citation correctness in KPI-RAG. The project preserves version history so changes in 3GPP interpretation and validation can be reviewed transparently.

This folder is therefore a reproducibility and traceability artifact for the standards-grounding layer.