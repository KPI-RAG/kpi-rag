# KPI-RAG: Full Repo Scaffold Plan

**Mode**: Direct | **Date**: 2026-08-07 | **Scope**: Ambiguity list -> Agent config stack -> Module specs -> Phase 0 execution plan

---

## STEP 1 -- AMBIGUITIES FLAGGED

Every underspecified or dual-interpretable implementation detail found in the proposal, with proposed resolutions. **No file that depends on an ambiguity will be finalized until confirmed.**

| # | Section | Ambiguity | Proposed Resolution | Blocks |
|---|---------|-----------|---------------------|--------|
| A1 | S6.3 L2 | **ClassifierOutput JSON schema not specified byte-for-byte.** Proposal says "structured JSON consumed by Layer 3" but only describes fields narratively. Exact field names, types, nesting unspecified. | Define canonical schema: sample_id: str, binary_prediction: int (0/1), binary_confidence: float, fault_type: str, fault_confidence: float, shap_attributions: array of {channel, value, direction}, protocol_state: {ul, dl}, feature_vector_dim: int. | schema-contracts, LLMExplainer, RAGQueryConstructor |
| A2 | S6.3 L3C | **LLMExplanation JSON schema not byte-for-byte.** Five fields named but types, required/optional, reasoning_trace type unspecified. | All five string, all required. reasoning_trace = single string. Add grounded: bool and generation_method: "llm" or "template" as metadata. | schema-contracts, Evaluator |
| A3 | S6.6 M7 | **KGIndexer vs. RAGQueryConstructor boundary.** Single module but two responsibilities. One file or two? | Two files: kg_indexer.py (ChromaDB indexing) + query_constructor.py (query template). Shared config. | Phase 0 file list |
| A4 | S6.3 | **ChromaDB client variant.** "Direct client" but PersistentClient vs Client vs HttpClient unspecified. | chromadb.PersistentClient(path=cfg.chroma_dir). No server mode. | KGIndexer |
| A5 | S6.3 | **Top-K for SHAP attributions.** "top-K" in L2 vs "top-3" in query template. | Store all 18 channels in ClassifierOutput. Query constructor uses top-3 by abs(SHAP). Dashboard shows configurable top-K (default 5). | SHAPExplainer, RAGQueryConstructor |
| A6 | S6.3 | **Window params config vs hardcoded.** 128/32/8 stated as facts. | Config-driven with those defaults. feature_version hash for invalidation. | DatasetBuilder, config |
| A7 | S6.3 | **80/20 rebalancing method.** SMOTE, oversampling, undersampling, or just class_weight? | Random undersample normal to match anomalous in train. class_weight='balanced' on top. | Detector |
| A8 | S6.6 M9 | **TeleQnA probe automated or manual?** G-Eval requires human reviewers. | Track A: automated. Track B: TeleQnA automated, G-Eval human-assisted. Track C: automated runs, human G-Eval. | Evaluator |
| A9 | S6.3 | **Alignment table JSON structure.** Field types and valid relation_types unspecified. | All string. relation_type enum: defines, specifies_threshold, describes_procedure, references. | AlignmentTable, LLMExplainer |
| A10 | S6.3 | **Similarity threshold 0.45 -- config or hardcoded?** | Config key rag.similarity_threshold defaulting to 0.45. | RAGQueryConstructor |
| A11 | S6.6 M10 | **Dashboard layout.** Single-page vs multi-tab? | Single-page Streamlit, sidebar for sample selection, 5 vertical panels. | Dashboard |
| A12 | S6.3 | **LLM model config structure.** "Interchangeable" but no config structure. | Config llm section with provider, model_name, temperature, max_tokens, json_mode. Provider-specific nested keys. | LLMExplainer |
| A13 | S7 | **G-Eval scoring -- simple mean or weighted?** | Equal weight, simple arithmetic mean across 4 dimensions. | evaluation-protocol, Evaluator |
| A14 | S6.3 | **Retry count before template fallback.** "Two failed retries" -- 2 or 3 total? | 3 total (1 initial + 2 retries). Template fallback after 3. | LLMExplainer |
| A15 | S6.3 | **Constrained decoding libs.** "outlines/guidance" -- both or OR? | outlines for Ollama. Groq native JSON mode for Groq. guidance as fallback only. | LLMExplainer |
| A16 | S6.5 | **TelecomTS data format.** JSONL, Parquet, or Arrow? | Load via datasets library from HuggingFace. Underlying JSONL. | DatasetBuilder |
| A17 | S6.3 | **Normal-class mean -- raw channel or feature?** | Mean of precomputed statistics features (mean per channel) on normal-class training samples. Stored as normal_means.npy. | SHAPExplainer, RAGQueryConstructor |
| A18 | S6.6 | **Module 7 ownership.** Proposal says P3, user says M7-10 is their scope. | Scaffold all 10. Flag ownership as 7-10 per user. Note P3/P4 discrepancy. | Module specs |

> [!IMPORTANT]
> **A1 and A2 are critical** -- the exact JSON schemas gate schema-contracts.md, the SHAP-JSON validation skill, and all of Layer 3. Please confirm or revise the proposed field sets before I finalize those files.

---

## STEP 2 -- AGENT CONFIG STACK

### File 1: /AGENTS.md

```markdown
# AGENTS.md -- Hard Constraints (Never-Violate Rules)

These rules are extracted verbatim or derived directly from the KPI-RAG proposal
Section 6.3 ("Technical Details"). Every agent, human or automated, must comply.
Violations must be caught in CI/pre-commit, not in code review.

## Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- No substitutions without full re-indexing and re-evaluation.
- Used for: ChromaDB ticket/QnA indexing AND retrieval query embedding.

## Vector Store
- **ChromaDB direct client** via chromadb.PersistentClient(path=...).
- **No LangChain. No LlamaIndex. No wrapper frameworks.**
- Collections managed directly through the ChromaDB Python API.

## Scale Preservation
- **No normalization on raw KPI values.** No StandardScaler, MinMaxScaler,
  or any transform that destroys absolute magnitude before feature extraction.
- Absolute-scale information must survive into the 582-dimensional vector.
- The only transforms applied are: re-computation of statistics from raw values,
  patchwise statistics, first-order differences, and one-hot encoding.

## Reproducibility
- random_state=42 for ALL stochastic operations: train/test split,
  RandomizedSearchCV, Random Forest, XGBoost, any sampling.
- Split indices saved as train_idx.npy / test_idx.npy and reused everywhere.

## Configuration
- **Config-driven values only.** No magic numbers in source code.
- All thresholds, paths, model names, hyperparameter search spaces, and
  feature dimensions must come from config/default.yaml or environment
  overrides.
- Config schema is validated at startup.

## Logging
- logging module only. **No print() statements** in any module.
- Logger name = module name (logging.getLogger(__name__)).
- One-line log messages. No start/finish log pairs.

## Inter-Layer JSON Schemas

### ClassifierOutput (Layer 2 -> Layer 3)

    {
      "sample_id": "string",
      "binary_prediction": 0,
      "binary_confidence": 0.0,
      "fault_type": "string",
      "fault_confidence": 0.0,
      "shap_attributions": [
        {"channel": "string", "value": 0.0, "direction": "above|below"}
      ],
      "protocol_state": {"ul": "TCP|UDP|None", "dl": "TCP|UDP|None"},
      "feature_vector_dim": 582
    }

### LLMExplanation (Layer 3 output)

    {
      "root_cause": "string",
      "3gpp_reference": "string",
      "oran_component": "string",
      "recommended_action": "string",
      "reasoning_trace": "string",
      "grounded": true,
      "generation_method": "llm|template"
    }

### AlignmentTableRow (static lookup)

    {
      "fault_type": "string",
      "document_id": "string",
      "release": "string",
      "clause": "string",
      "evidence_span": "string",
      "relation_type": "defines|specifies_threshold|describes_procedure|references",
      "oran_component": "string"
    }

## Excluded Fields
- TelecomTS fields statistics, anomalies, labels, QnA, description,
  and troubleshooting_tickets are **explicitly excluded from classifier inputs**.
- Only the KPIs field is used for feature extraction.
- Tickets and QnA are used only in Layer 3 (RAG corpus).
```

---

### File 2: /GEMINI.md

```markdown
# GEMINI.md -- KPI-RAG Project Blueprint for Antigravity

## What Is KPI-RAG?

KPI-RAG is an end-to-end pipeline for explainable root-cause analysis of 5G
network faults. It takes raw KPI time series as input and produces
standards-grounded natural-language explanations as output, displayed in a
Streamlit dashboard.

## Three-Layer Architecture

1. **Layer 1 -- Preprocessing and Feature Engineering**: Loads TelecomTS dataset,
   constructs 582-dimensional feature vectors per sample (64 statistics +
   256 patchwise scale stats + 256 first-order differences + 6 categorical
   encodings), produces stratified train/test split.

2. **Layer 2 -- Detection, Classification and SHAP**: Two cascaded Random Forest
   models (binary anomaly detection then 11-class fault classification) with
   SHAP TreeExplainer producing per-channel attributions. Outputs structured
   JSON consumed by Layer 3.

3. **Layer 3 -- RAG and Standards-Grounded Explanation**: Retrieves similar
   historical tickets from ChromaDB, injects pre-validated 3GPP clause from
   the Alignment Table, generates structured explanation via LLM with
   constrained decoding.

## My Role and Scope

I am P4, implementing Modules 7-10:
- **Module 7**: KGIndexer + RAGQueryConstructor (ChromaDB indexing + SHAP-derived queries)
- **Module 8**: LLMExplainer (prompt fusion, constrained decoding, citation validation)
- **Module 9**: Evaluator (Track A detection metrics, Track B G-Eval, Track C ablation)
- **Module 10**: Dashboard (Streamlit 5-panel app, HuggingFace Spaces deployment)

I am scaffolding the full repo (all 10 modules) since Phase 0 covers shared
infrastructure, but Modules 1-6 are owned by P1/P2/P3.

## Dataset

**TelecomTS** [D1]: 32,000 samples, 18 KPI channels (16 numerical +
2 categorical), 128 timesteps at 100ms, 11 anomaly types + normal class.
1,235 anomalous samples with troubleshooting tickets and QnA reasoning traces.
MIT license, hosted on HuggingFace.

## Research Questions

- **RQ1**: Does scale-preserving feature engineering (582-dim) improve anomaly
  detection F1 and fault classification F1-macro vs. normalized-feature and
  statistics-only baselines?
- **RQ2**: Does alignment-table grounding produce measurably higher citation
  faithfulness vs. ticket-only and ungrounded LLM generation?
- **RQ3**: Which feature configuration (full 582-dim, 326-dim without patchwise
  scale, or 64-dim statistics-only) offers the best performance-to-cost tradeoff?

## Success Criteria

| Objective | Criterion |
|-----------|-----------|
| O1 -- Anomaly Detection | Binary F1 comparable to Mantis (0.800); per-class F1-macro for 11-class |
| O2 -- Alignment Table | 10/10 rows validated, zero unresolved disputes |
| O3 -- Standards Retrieval | Acceptable precision@k on held-out queries (Week 9) |
| O4 -- Grounded Explanation | >=70% pass regex-validated 3GPP clause check |
| O5 -- Evaluation | Joint detection + explanation faithfulness protocol executed |

## Hard Rules

See AGENTS.md for all never-violate constraints.

## Environment

- **OS**: Windows, PowerShell
- **Python**: 3.10+
- **Testing**: pytest with markers (unit, integration, slow)
- **Dev compute**: Local CPU for Layers 1-2; Frontenac HPC for Layer 3
- **Demo**: HuggingFace Spaces via Groq API
```

---

### File 3: /.agents/rules/code-style.md

```markdown
# Code Style Rules

Human-written-in-one-sitting style. The code should look like it was written
by someone who knows exactly what they are doing and does not need to explain
the obvious.

## Naming
- Short names: cfg, db, vecs, tickets, preds, attrs, feat.
- Module-level abbreviations fine when unambiguous.
- No _manager, _handler, _service suffixes unless genuinely managing state.

## Documentation
- No verbose docstrings on obvious functions.
- Docstrings only when behavior is non-obvious, has side effects, or complex return.
- No "Initialize the..." on __init__.

## Comments
- No "# Initialize the logger" comments.
- Comments explain why, never what.
- Inline comments only for genuinely tricky logic.

## Error Handling
- No blanket try/except Exception blocks.
- Catch specific exceptions. Let unexpected errors propagate.
- Validation errors: ValueError with one-line message.

## Structure
- **Functions over classes** unless object genuinely holds state across calls.
- If a class has only __init__ and one method, make it a function.
- One module = one responsibility.

## Logging
- One-line messages: log.info("indexed %d tickets", n).
- No start/finish log pairs.
- Levels: debug=intermediate, info=milestones, warning=recoverable, error=failures.

## Imports
- stdlib, blank, third-party, blank, local.
- No wildcard imports.
- Prefer from pathlib import Path.

## Testing
- Mirror source: src/kpi_rag/foo.py -> tests/test_foo.py.
- Fixtures in conftest.py.
- assert x == y, not self.assertEqual.
```

---

### File 4: /.agents/rules/schema-contracts.md

```markdown
# Schema Contracts

Exact JSON schemas for contract tests. Any module producing or consuming
these structures must conform byte-for-byte.

## ClassifierOutput (Layer 2 -> Layer 3)

Schema file: src/kpi_rag/schemas/classifier_output.schema.json

Required fields:
- sample_id (string): Unique window identifier
- binary_prediction (integer, enum [0, 1]): 0=normal, 1=anomaly
- binary_confidence (number, 0.0-1.0): Uncalibrated RF probability
- fault_type (string): Predicted fault label or "normal"
- fault_confidence (number, 0.0-1.0): Uncalibrated RF probability
- shap_attributions (array of objects): All 18 channels, sorted by |value| desc
  - channel (string): KPI channel name
  - value (number): SHAP value (signed)
  - direction (string, enum ["above", "below"]): vs normal-class mean
- protocol_state (object): {ul: TCP|UDP|None, dl: TCP|UDP|None}
- feature_vector_dim (integer): 582, 326, or 64

additionalProperties: false

## LLMExplanation (Layer 3 output)

Schema file: src/kpi_rag/schemas/llm_explanation.schema.json

Required fields:
- root_cause (string)
- 3gpp_reference (string): e.g. "TS 38.300 clause 9.2.3"
- oran_component (string)
- recommended_action (string)
- reasoning_trace (string): Multi-sentence chain
- grounded (boolean)
- generation_method (string, enum ["llm", "template"])

additionalProperties: false

## AlignmentTableRow

Schema file: src/kpi_rag/schemas/alignment_row.schema.json

Required fields:
- fault_type (string)
- document_id (string): 3GPP TS document ID
- release (string)
- clause (string)
- evidence_span (string): Quoted spec text
- relation_type (string, enum ["defines", "specifies_threshold", "describes_procedure", "references"])
- oran_component (string)

additionalProperties: false

## Contract Test Requirements

- Every JSON blob crossing a layer boundary MUST validate before downstream accepts.
- jsonschema.validate() with Draft7Validator.
- Schema files in src/kpi_rag/schemas/ as .json.
- Contract tests in tests/contract/, run on every CI push.
```

---

### File 5: /.agents/rules/evaluation-protocol.md

```markdown
# Evaluation Protocol

Single source of truth. No future agent may invent a different rubric.

## Track A -- Detection Metrics (Automated)

- Model 1 (Binary): P/R/F1 on held-out test (natural ~96/4% distribution).
  Baseline: Mantis F1=0.800 (Table 2). Not like-for-like.
- Model 2 (11-class): Per-class F1, F1-macro, accuracy on anomalous-only.
  Context: Toto acc=0.848, Mantis acc=0.590 (Table 4).
- Cascade: Full 12-class confusion matrix over ALL held-out samples.
- Conditional: Model1 recall x Model2 accuracy on true anomalies.
- Jamming probe: 279 samples excluded from training; zero-exposure test.
- Small-class: Congestion-Sudden (n=48), Resource-Bugs (n=60) use
  stratified 5-fold CV with mean +/- 95% CI.

## Track B -- Explanation Quality

### G-Eval Rubric (4 Dimensions, equal weight)

| Dimension | Definition | Scale |
|-----------|-----------|-------|
| Citation Validity | Cited 3GPP TS exists, valid format (series 21-38), in alignment table | 1-5 |
| Fault Specificity | Correct fault type AND key KPI signatures match SHAP | 1-5 |
| Actionability | Recommended step supported by cited clause and evidence | 1-5 |
| Causal Soundness | Fault mechanism consistent with cited clause and KPI pattern | 1-5 |

- Overall = arithmetic mean of 4 dimensions.
- Set: 30 stratified explanations (3 per synthetic fault, jamming excluded).
- Evaluators: Two independent, external, blinded. 30-min calibration.
  Third adjudicator. Inter-rater agreement per dimension.

### Objective Citation Metric
- % passing: (1) format regex TS\s+[23][0-9]\.\d{3}, (2) alignment table lookup.
- Threshold: >=70%.
- Caveat: n=30, 95% CI ~+/-16pp. Directional pilot evidence.

### TeleQnA (Secondary, Automated)
- RAG accuracy on subset. Proxy only.

## Track C -- Three-Condition Ablation

Same model, prompt, settings. Only retrieval context differs:

| Cond | Label | Context |
|------|-------|---------|
| C1 | Label-only | Fault type only |
| C2 | Label+tickets | Fault type + top-k tickets/QnA |
| C3 | Full | Fault type + tickets/QnA + alignment row |

Core: C2 vs C3 isolates standards-grounding.
Scored on same 30 samples via G-Eval.
Contingency: drop C1 if Wk11 slips.

## Feature Ablation (RQ1/RQ3)

1. Full: 582-dim
2. No-scale: 326-dim
3. Stats-only: 64-dim
```

---

### File 6: /.agents/skills/3gpp-citation-check/SKILL.md

```markdown
---
name: 3gpp-citation-check
description: >
  Validate a 3GPP TS reference string for format correctness and
  cross-check against the alignment table.
---

# 3GPP Citation Check

## Purpose
Validate 3gpp_reference through two checks per proposal Section 6.3.

## Inputs
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| reference | string | Yes | 3GPP reference to validate |
| alignment_table_path | string | No | Path to alignment_table.json |

## Check 1: Format Regex
Pattern: ^TS\s+([23][0-9])\.(\d{3})\b
- Series 21-38. Three-digit sub-number. Clause ref optional.

## Check 2: Alignment Table Cross-Reference
- Load alignment_table.json. Extract document_id set. Check membership.

## Output
| Field | Type | Description |
|-------|------|-------------|
| valid | bool | Both checks pass |
| format_ok | bool | Check 1 |
| in_alignment_table | bool | Check 2 |
| extracted_ts | string/null | Extracted TS number |
| hallucinated_but_valid_format | bool | Format ok but not in table |
| error | string/null | Error message |

## Usage
Invoke on every LLMExplanation.3gpp_reference before setting grounded=true.
```

---

### File 7: /.agents/skills/shap-json-validate/SKILL.md

```markdown
---
name: shap-json-validate
description: >
  Validate a Layer 2 ClassifierOutput JSON blob against the canonical
  schema before it enters Layer 3.
---

# SHAP JSON Validation

## Purpose
Gate-check ClassifierOutput before RAGQueryConstructor or LLMExplainer.

## Inputs
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| blob | object | Yes | ClassifierOutput JSON |
| schema_path | string | No | Path to schema file |

## Validation
1. **Schema**: jsonschema.validate() with Draft7Validator.
2. **Semantic**:
   - 18 shap_attributions entries
   - Sorted by |value| descending
   - binary_prediction=0 implies fault_type="normal"
   - binary_prediction=1 implies fault_type!="normal"
   - feature_vector_dim in {582, 326, 64}
   - Confidence in [0.0, 1.0]
3. **Channel names**: All 18 in canonical list.

## Output
| Field | Type | Description |
|-------|------|-------------|
| valid | bool | All checks pass |
| errors | string[] | Errors |
| warnings | string[] | Non-fatal issues |

## Usage
Call before build_query(), generate(), or eval output writes.
Failures raise ValueError; sample marked validation_failed.
```

---

## STEP 3 -- MODULE SPECS

| # | Module (Owner) | Inputs | Outputs | Acceptance Criteria | Deps | Phase | Crit? |
|---|---------------|--------|---------|---------------------|------|-------|-------|
| 1 | DatasetBuilder (P1) | TelecomTS HF | X/y .npy, split idx | shape[1]=582; 12 classes; no NaN; rs=42; raw scale; idx disjoint | Config | Ph3 W6-7 | Yes |
| 2 | ScaleAblation (P1) | M1 outputs | 3 triples (582/326/64) | 326 removes 256 patchwise; 64 keeps first 64; labels same | M1 | Ph3 W7 | No |
| 3 | Detector (P2) | X_train, y_binary | Model1 joblib, metrics | F1 computed; balanced; rs=42; IsoForest baseline; StratCV | M1 | Ph3 W7 | Yes |
| 4 | Classifier (P2) | X_anom, y_multi | Model2 joblib, F1 table | 11 classes; 11-row F1; strat k-fold n<100; XGBoost path | M1 | Ph3 W7-8 | Yes |
| 5 | SHAPExplainer (P2) | Models, test | SHAP, normal_means, JSON | Shape match; schema valid; direction; 18 channels | M3,M4 | Ph3 W8 | Yes |
| 6 | AlignmentTable (P3) | Specs, KGs | JSON (10 rows) | 10 rows; schema; regex; no dup; valid enum | None | Ph2-3 W3-6 | Yes |
| 7 | KGIndexer+RAGQuery (P3/P4) | Tickets, JSON | ChromaDB, query, top-k | 1235 docs; MiniLM; no normal; query template; threshold | M1,M5 | Ph3 W8-9 | Yes |
| 8 | LLMExplainer (P4) | JSON, tickets, align | Explanation JSON | Schema; citation check; 3 retries; method flag; jamming | M6,M7 | Ph3 W9-11 | Yes |
| 9 | Evaluator (P4) | All outputs | Track A/B/C results | 12x12 CM; G-Eval 1-5; 3 conds; CI; jamming separate | M3-5,M8 | Ph4 W12-13 | End |
| 10 | Dashboard (P4) | Pipeline outputs | Streamlit, HF Spaces | Launches; 5 panels; selector; normal msg; SHAP; sources | M5,M7,M8 | Ph3-4 W11 | Demo |

### Critical Path
```
DatasetBuilder -> Detector -> SHAPExplainer -> KGIndexer -> LLMExplainer -> Evaluator
                  Classifier ----------------------------------------^
AlignmentTable (parallel) --------------------------> LLMExplainer
Dashboard (parallel after Wk9)
```

### Team Parallelization
| Person | Parallel (Wk 6-8) | Sequential (Wk 9-11) |
|--------|-------------------|----------------------|
| P1 | DatasetBuilder -> ScaleAblation | Ablation re-runs |
| P2 | Detector -> Classifier -> SHAPExplainer | Cascade eval |
| P3 | AlignmentTable -> KGIndexer | RAG calibration |
| P4 | Config/scaffold, Dashboard skeleton | LLMExplainer -> Evaluator -> Dashboard |

---

## STEP 4 -- PHASE 0 EXECUTION PLAN (Scaffold Only)

### 0.1 -- Project Root

| # | File | Purpose | Assertions |
|---|------|---------|------------|
| 1 | pyproject.toml | Metadata, deps, pytest | Valid TOML; name=kpi-rag; all deps |
| 2 | config/default.yaml | Config defaults | Valid YAML; 9 sections; rs=42; wl=128; s=32; pc=8; MiniLM; 0.45 |
| 3 | config/schema.yaml | Validation schema | Every key covered; passes on defaults |

### 0.2 -- Package

| # | File | Purpose | Assertions |
|---|------|---------|------------|
| 4 | src/kpi_rag/__init__.py | Package, version | Import ok; semver |
| 5 | src/kpi_rag/config.py | Load/validate config | Returns dict; rs=42; ValueError on missing; env override |
| 6 | src/kpi_rag/schemas/__init__.py | Schema loader | load_schema works; has $schema |
| 7 | schemas/classifier_output.schema.json | ClassifierOutput | Valid JSON; good passes; missing sample_id fails |
| 8 | schemas/llm_explanation.schema.json | LLMExplanation | Valid JSON; good passes; bad method fails |
| 9 | schemas/alignment_row.schema.json | AlignmentTableRow | Valid JSON; good passes; bad relation_type fails |
| 10 | src/kpi_rag/validate.py | Validation wrappers | Valid passes; invalid raises; extra fields rejected |

### 0.3 -- Module Stubs

| # | File | Purpose | Assertions |
|---|------|---------|------------|
| 11 | src/kpi_rag/dataset_builder.py | build_features stub | Callable; 6 keys; NotImplementedError |
| 12 | src/kpi_rag/scale_ablation.py | ablate stub | full=(10,582); no_scale=(10,326); stats=(10,64) |
| 13 | src/kpi_rag/detector.py | train_detector stub | Callable; NotImplementedError |
| 14 | src/kpi_rag/classifier.py | train_classifier stub | Callable; NotImplementedError |
| 15 | src/kpi_rag/shap_explainer.py | explain stub | Callable; NotImplementedError |
| 16 | src/kpi_rag/alignment_table.py | load/lookup stub | Callable; None for unknown |
| 17 | src/kpi_rag/kg_indexer.py | index_tickets stub | Callable; NotImplementedError |
| 18 | src/kpi_rag/query_constructor.py | build_query stub | Callable; NotImplementedError |
| 19 | src/kpi_rag/llm_explainer.py | generate_explanation stub | Callable; NotImplementedError |
| 20 | src/kpi_rag/evaluator.py | track_a/b/c stubs | Callable; NotImplementedError |
| 21 | src/kpi_rag/dashboard.py | Streamlit stub | Importable; no print; uses logging |
| 22 | src/kpi_rag/citation_check.py | FULL citation validation | valid TS passes; bad series fails; hallucination detected |

### 0.4 -- Tests

| # | File | Purpose | Assertions |
|---|------|---------|------------|
| 23 | tests/conftest.py | Fixtures | All return schema-valid data |
| 24 | tests/contract/__init__.py | Package | Importable |
| 25 | tests/contract/test_classifier_output.py | Contract | Valid/missing/extra/bad-enum |
| 26 | tests/contract/test_llm_explanation.py | Contract | Valid/bad-method/missing-grounded |
| 27 | tests/contract/test_alignment_row.py | Contract | Valid/bad-relation |
| 28 | tests/test_config.py | Config | Loads/missing/invalid |
| 29 | tests/test_citation_check.py | Citation | Valid/bad-series/hallucination/boundaries |
| 30 | tests/test_validate.py | Validation | Valid/invalid/extra |

### 0.5 -- Meta

| # | File | Purpose |
|---|------|---------|
| 31 | (pytest markers in pyproject.toml) | unit, integration, slow, contract |
| 32 | .gitignore | __pycache__, .venv, *.npy, *.joblib, chroma_db/ |
| 33 | README.md | Overview, setup, module map |
| 34 | data/.gitkeep | TelecomTS placeholder |
| 35 | models/.gitkeep | Model artifacts |
| 36 | results/.gitkeep | Eval outputs |

### Phase 0 Completion Gate

```
pytest -m contract --tb=short
pytest tests/test_config.py
pytest tests/test_citation_check.py
pytest tests/test_validate.py
python -c "import kpi_rag"
```

**36 files** (22 source + 9 test + 5 meta). Zero ML logic. ~2-3 hours for execution agent.

---

## Execution Handoff

```text
Launch via /omagy:ralph "Implement KPI-RAG Phase 0 scaffold per the PRD -- file-by-file, pytest after each, stop at Phase 0 completion gate"
```

> [!IMPORTANT]
> Before executing, confirm the 18 ambiguity resolutions in Step 1, especially **A1** and **A2** which gate the schema contract files.
