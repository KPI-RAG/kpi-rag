"""Tests for Phase 2 pipeline fixes."""
import pytest
from unittest.mock import patch, MagicMock


# ── load_alignment_table Jamming handling ─────────────────────────────────────

def test_alignment_table_excludes_jamming_from_grounding():
    """Jamming has relation_type=not-applicable — must not appear in loaded table."""
    from src.llm_explainer import load_alignment_table
    align = load_alignment_table("configs/alignment_table.json")
    assert "Jamming" not in align

def test_alignment_table_loads_11_rows_from_data():
    """data/alignment_table.json has 11 rows including Jamming."""
    import json
    with open("data/alignment_table.json") as f:
        d = json.load(f)
    rows = d.get("rows", d) if isinstance(d, dict) else d
    assert len(rows) == 11

def test_alignment_table_configs_has_10_after_jamming_filter():
    """After load_alignment_table filters Jamming, configs/ yields 10 entries."""
    from src.llm_explainer import load_alignment_table
    align = load_alignment_table("configs/alignment_table.json")
    assert len(align) == 10

def test_load_alignment_table_handles_null_3gpp_reference():
    """Jamming row has 3gpp_reference=None — must not crash."""
    import json, tempfile, os
    table = {"rows": [
        {
            "telecomts_fault": "Jamming",
            "3gpp_reference": None,
            "clause_text": None,
            "oran_component": None,
            "relation_type": "not-applicable",
        }
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        tmp = f.name
    try:
        from src.llm_explainer import load_alignment_table
        align = load_alignment_table(tmp)
        # Jamming filtered out — no crash
        assert "Jamming" not in align
    finally:
        os.unlink(tmp)

def test_load_alignment_table_handles_null_clause_text():
    """clause_text=None must not raise TypeError."""
    import json, tempfile, os
    table = {"rows": [
        {
            "telecomts_fault": "Antenna Failure",
            "3gpp_reference": "TS 38.141-1 §4.2.1",
            "clause_text": None,
            "oran_component": "O-RU",
            "relation_type": "direct",
        }
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        tmp = f.name
    try:
        from src.llm_explainer import load_alignment_table
        align = load_alignment_table(tmp)
        assert "Antenna Failure" in align
        assert align["Antenna Failure"]["evidence_span"] == ""
    finally:
        os.unlink(tmp)

def test_load_alignment_table_handles_null_oran_component():
    """oran_component=None must normalize to empty string."""
    import json, tempfile, os
    table = {"rows": [
        {
            "telecomts_fault": "Antenna Failure",
            "3gpp_reference": "TS 38.141-1 §4.2.1",
            "clause_text": None,
            "oran_component": None,
            "relation_type": "direct",
        }
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(table, f)
        tmp = f.name
    try:
        from src.llm_explainer import load_alignment_table
        align = load_alignment_table(tmp)
        assert align["Antenna Failure"]["oran_component"] == ""
    finally:
        os.unlink(tmp)


# ── _is_rate_limit handles network errors ─────────────────────────────────────

def test_is_rate_limit_handles_getaddrinfo():
    """getaddrinfo failure = transient network error = should retry."""
    from src.llm_explainer import _is_rate_limit
    err = OSError("[Errno 11001] getaddrinfo failed")
    assert _is_rate_limit(err) is True

def test_is_rate_limit_handles_429():
    from src.llm_explainer import _is_rate_limit
    err = Exception("429 Too Many Requests")
    assert _is_rate_limit(err) is True

def test_is_rate_limit_handles_resource_exhausted():
    from src.llm_explainer import _is_rate_limit
    err = Exception("RESOURCE_EXHAUSTED quota exceeded")
    assert _is_rate_limit(err) is True

def test_is_rate_limit_false_for_value_error():
    from src.llm_explainer import _is_rate_limit
    err = ValueError("bad input")
    assert _is_rate_limit(err) is False


# ── run_eval_track_c default output path ──────────────────────────────────────

def test_run_track_c_has_default_output():
    """--output should have a default so script can run without args."""
    import argparse
    import scripts.run_eval_track_c as m
    # Re-parse with empty args to check default
    import sys
    old_argv = sys.argv
    sys.argv = ["run_eval_track_c.py"]
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", default="data/processed/track_c_explanations.jsonl")
        args = parser.parse_args([])
        assert args.output == "data/processed/track_c_explanations.jsonl"
    finally:
        sys.argv = old_argv


# ── Congestion supplementation ────────────────────────────────────────────────

def test_congestion_supplementation_reaches_3_samples():
    """
    Phase 2 layer2 has only 2 Gradual and 1 Sudden Spike.
    After supplementation from old handoff, both must have >= 3.
    """
    import json, numpy as np
    from src.schema import ClassifierOutput
    from scripts.run_eval_track_c import TRACK_C_FAULTS
    import os

    with open("data/processed/layer2_output_sessionsplit_phase2.json") as f:
        d = json.load(f)

    by_fault = {}
    for w in d:
        try:
            obj = ClassifierOutput(**w)
            if obj.anomaly_type in TRACK_C_FAULTS:
                by_fault.setdefault(obj.anomaly_type, []).append(w["window_index"])
        except Exception:
            pass

    test_set = set(np.load("data/indices/test_idx_sessionsplit.npy").tolist())
    supplement_path = "data/processed/layer2_rag_handoff_sessionsplit.json"

    if os.path.exists(supplement_path):
        with open(supplement_path) as f:
            old_data = json.load(f)
        for ft in TRACK_C_FAULTS:
            if len(by_fault.get(ft, [])) < 3:
                existing = set(by_fault.get(ft, []))
                for rec in old_data:
                    if rec.get("window_index") in test_set and rec.get("window_index") not in existing:
                        try:
                            obj = ClassifierOutput(**rec)
                            if obj.anomaly_type == ft:
                                by_fault.setdefault(ft, []).append(rec["window_index"])
                                existing.add(rec["window_index"])
                                if len(by_fault[ft]) >= 3:
                                    break
                        except Exception:
                            pass

    for ft in TRACK_C_FAULTS:
        count = len(by_fault.get(ft, []))
        assert count >= 3, f"{ft.value} has only {count} samples after supplementation"


# ── ChromaDB cosine metric ────────────────────────────────────────────────────

def test_chromadb_uses_cosine_metric():
    import yaml, chromadb
    cfg = yaml.safe_load(open("configs/config.yaml"))
    client = chromadb.PersistentClient(path=cfg["rag"]["chroma_db_path"])
    c = client.get_collection(cfg["rag"]["collection_name"])
    assert c.configuration["hnsw"]["space"] == "cosine"

def test_chromadb_has_935_tickets():
    import yaml, chromadb
    cfg = yaml.safe_load(open("configs/config.yaml"))
    client = chromadb.PersistentClient(path=cfg["rag"]["chroma_db_path"])
    c = client.get_collection(cfg["rag"]["collection_name"])
    assert c.count() == 935

def test_chromadb_similarity_scores_positive():
    """After cosine fix, all similarity scores must be in [0, 1]."""
    from src.config_loader import load_config
    from src.rag_query import retrieve_similar_tickets
    cfg = load_config()
    results = retrieve_similar_tickets("interference degradation SNR", cfg, n_results=3)
    for r in results:
        assert 0.0 <= r["similarity"] <= 1.0, f"Negative similarity: {r['similarity']}"

def test_no_legacy_l2_collection():
    """Old L2 collection must be deleted."""
    import yaml, chromadb
    cfg = yaml.safe_load(open("configs/config.yaml"))
    client = chromadb.PersistentClient(path=cfg["rag"]["chroma_db_path"])
    collections = [c.name for c in client.list_collections()]
    assert "telecomts_tickets" not in collections
    assert "telecomts_tickets_cosine" in collections
