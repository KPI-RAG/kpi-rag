# Test Spec: KPI-RAG Phase 0 Scaffold

**PRD**: prd-20260807T153653Z-kpi-rag-scaffold.md

## Verification Matrix

| Risk | Check | Type | Expected Result |
|------|-------|------|-----------------|
| Config schema drift | pytest tests/test_config.py | unit | default.yaml loads; all 9 sections present; random_state=42 |
| Config env override broken | pytest tests/test_config.py::test_env_override | unit | KPI_RAG_DATA__RANDOM_STATE=99 overrides YAML |
| ClassifierOutput schema mismatch | pytest -m contract tests/contract/test_classifier_output.py | contract | Valid passes; missing sample_id fails; extra field fails; bad direction fails |
| LLMExplanation schema mismatch | pytest -m contract tests/contract/test_llm_explanation.py | contract | Valid passes; bad generation_method fails; missing grounded fails |
| AlignmentTableRow schema mismatch | pytest -m contract tests/contract/test_alignment_row.py | contract | Valid passes; bad relation_type fails |
| Citation check false positive | pytest tests/test_citation_check.py::test_valid_ref | unit | "TS 38.300 clause 9.2.3" returns valid=True |
| Citation check false negative | pytest tests/test_citation_check.py::test_invalid_series | unit | "TS 99.999" returns format_ok=False |
| Hallucination not detected | pytest tests/test_citation_check.py::test_hallucinated | unit | Valid format but not in table returns hallucinated=True |
| Citation boundary series | pytest tests/test_citation_check.py::test_boundaries | unit | 21.xxx ok; 20.xxx fail; 38.xxx ok; 39.xxx fail |
| Schema validation wrappers broken | pytest tests/test_validate.py | unit | Valid passes; invalid raises ValidationError; extra rejected |
| Package not importable | python -c "import kpi_rag" | static | Exit code 0 |
| Module stubs missing functions | pytest tests/test_stubs.py (via conftest discovery) | unit | All stub functions callable |
| Scale ablation dimensions wrong | pytest tests/test_scale_ablation_stub.py | unit | full=(n,582); no_scale=(n,326); stats=(n,64) |
| Contract tests not discoverable | pytest --co -m contract | static | Discovers 3 contract test files |
| No print() in source | grep -r "print(" src/kpi_rag/ (exclude tests) | static | Zero matches |

## Phase 0 Completion Gate

All of the following must pass before any Phase 1+ work begins:

```bash
pytest -m contract --tb=short        # 3 contract test files
pytest tests/test_config.py          # Config system
pytest tests/test_citation_check.py  # Citation validation
pytest tests/test_validate.py        # Schema wrappers
python -c "import kpi_rag"           # Package import
```

## Scope

- Phase 0 only (scaffold). No ML logic, no model training, no data loading.
- citation_check.py is the only fully-implemented module.
- All other modules are stubs raising NotImplementedError.
