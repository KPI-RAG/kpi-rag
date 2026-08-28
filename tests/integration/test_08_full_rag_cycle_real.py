"""Test 08 — Full RAG cycle: real ChromaDB + real LLM (skipped if unavailable)."""
import os
import pytest
from src.config_loader import load_config
from src.schema import ClassifierOutput, AnomalyType, LLMExplanation
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain, validate_citation
from src.utils import validate_3gpp_ref

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
HAS_GROQ = bool(GROQ_KEY)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def collection(cfg):
    col = get_collection(cfg)
    if col.count() == 0:
        pytest.skip("ChromaDB empty — run build_index.py first")
    return col


@pytest.fixture(scope="module")
def alignment():
    return load_alignment_table("configs/alignment_table.json")


# ─── TEST 1: Alignment table completeness ──────────────────────────────────────
def test_alignment_all_10_faults_resolve(alignment):
    """All 10 non-Jamming faults have valid entries with required fields."""
    for fault in AnomalyType:
        if fault.value == "Jamming":
            assert fault.value not in alignment
            continue
        assert fault.value in alignment, f"MISSING: {fault.value}"
        entry = alignment[fault.value]
        assert entry.get("3gpp_ts"), f"Empty 3gpp_ts for {fault.value}"
        assert entry.get("clause"), f"Empty clause for {fault.value}"
        assert entry.get("oran_component"), f"Empty oran_component for {fault.value}"
        # Rodina's rich fields must be preserved by load_alignment_table()
        assert "causal_mechanism" in entry, (
            f"causal_mechanism missing for {fault.value}"
        )


# ─── TEST 2: RAG retrieval — all 11 fault types ────────────────────────────────
def test_retrieval_all_11_faults(all_payloads, collection, cfg):
    """Real ChromaDB retrieval for all 11 fault types must not crash."""
    results = {}
    for fault_name, payload_data in all_payloads.items():
        payload = ClassifierOutput(**payload_data)
        tickets, low_conf = query_from_classifier_output(
            payload, collection, cfg
        )
        assert isinstance(tickets, list)
        assert isinstance(low_conf, bool)
        best_sim = max((t.similarity_score for t in tickets), default=0.0)
        results[fault_name] = {
            "n_tickets": len(tickets),
            "best_sim": best_sim,
            "low_conf": low_conf,
        }
        if tickets:
            for t in tickets:
                assert -0.1 <= t.similarity_score <= 1.1, (
                    f"{fault_name}: similarity {t.similarity_score} out of range"
                )

    print("\n  Retrieval results:")
    for fault, r in results.items():
        print(
            f"    {fault}: n={r['n_tickets']} "
            f"best_sim={r['best_sim']:.3f} low_conf={r['low_conf']}"
        )


# ─── TEST 3: Full explain — Antenna Failure ────────────────────────────────────
def test_full_explain_antenna_failure(collection, cfg, alignment):
    """Full pipeline with real LLM — Antenna Failure."""
    if not HAS_GROQ:
        pytest.skip("GROQ_API_KEY not set")

    payload = ClassifierOutput(**{
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"},
        ],
        "signal_statistics": {
            "RSRP_mean": -108, "RSRP_std": 4.2, "RSRP_min": -115, "RSRP_max": -98,
            "DL_BLER_mean": 0.38, "DL_BLER_std": 0.09, "DL_BLER_min": 0.22, "DL_BLER_max": 0.54,
        },
    })

    tickets, low_conf = query_from_classifier_output(payload, collection, cfg)
    result = explain(payload, tickets, cfg, alignment)

    assert isinstance(result, LLMExplanation)
    assert result.root_cause
    assert result.oran_component
    assert isinstance(result.reference_valid, bool)
    assert isinstance(result.template_generated, bool)

    print(f"\n  Antenna Failure result:")
    print(f"    root_cause: {result.root_cause}")
    print(
        f"    3gpp_ref:   {result.gpp_reference} "
        f"valid={result.reference_valid}"
    )
    print(f"    oran:       {result.oran_component}")
    print(f"    tickets:    {len(tickets)}")
    print(f"    template:   {result.template_generated}")


# ─── TEST 4: Citation validity across 5 fault types ────────────────────────────
def test_citation_validity_5_faults(all_payloads, collection, cfg, alignment):
    """Citation valid rate >= 60% across 5 fault types with Rodina's alignment table."""
    if not HAS_GROQ:
        pytest.skip("GROQ_API_KEY not set")

    test_faults = [
        "Antenna Failure",
        "Buffer Overflow (Gradual Buildup)",
        "Co-Channel Interference (Mild)",
        "Faulty RF Filters (Temporal)",
        "High Network Congestion (Sudden Spike)",
    ]

    valid_count = 0
    results = []

    for fault_name in test_faults:
        payload = ClassifierOutput(**all_payloads[fault_name])
        tickets, _ = query_from_classifier_output(payload, collection, cfg)
        result = explain(payload, tickets, cfg, alignment)
        results.append({
            "fault": fault_name,
            "ref": result.gpp_reference,
            "valid": result.reference_valid,
            "template": result.template_generated,
        })
        if result.reference_valid:
            valid_count += 1

    rate = valid_count / len(test_faults)
    print(f"\n  Citation validity: {valid_count}/{len(test_faults)} = {rate:.0%}")
    for r in results:
        icon = "\u2713" if r["valid"] else "\u2717"
        tmpl = " [TEMPLATE]" if r["template"] else ""
        print(f"    {icon} {r['fault']}: {r['ref']}{tmpl}")

    assert rate >= 0.60, (
        f"Citation validity {rate:.0%} below 60% — check LLM + alignment table"
    )


# ─── TEST 5: Jamming — no alignment entry ──────────────────────────────────────
def test_jamming_no_alignment_entry(all_payloads, collection, cfg, alignment):
    """Jamming has no alignment entry — pipeline must not crash; reference_valid=False is correct."""
    if not HAS_GROQ:
        pytest.skip("GROQ_API_KEY not set")

    payload = ClassifierOutput(**all_payloads["Jamming"])
    tickets, _ = query_from_classifier_output(payload, collection, cfg)
    result = explain(payload, tickets, cfg, alignment)

    assert isinstance(result, LLMExplanation)
    # Jamming has no alignment entry → reference_valid=False is CORRECT
    print(f"\n  Jamming result:")
    print(
        f"    ref={result.gpp_reference} "
        f"valid={result.reference_valid} "
        f"template={result.template_generated}"
    )
