"""Test 04 — Alignment table correctness and completeness."""
import pytest
from collections import defaultdict
from src.llm_explainer import load_alignment_table
from src.utils import validate_3gpp_ref


@pytest.fixture(scope="module")
def alignment():
    return load_alignment_table("configs/alignment_table.json")


def test_alignment_table_loads(alignment):
    """Alignment table must load as a dict with 10 entries."""
    assert isinstance(alignment, dict)
    assert len(alignment) == 10


def test_jamming_excluded(alignment):
    """Jamming must not be in the alignment table."""
    assert "Jamming" not in alignment


def test_all_10_faults_present(alignment):
    """All 10 non-Jamming fault types must be present."""
    expected = [
        "Antenna Failure",
        "Buffer Overflow (Gradual Buildup)",
        "Co-Channel Interference (Mild)",
        "Co-Channel Interference (Severe)",
        "Faulty RF Filters (Temporal)",
        "High Network Congestion (Gradual Buildup)",
        "High Network Congestion (Sudden Spike)",
        "Doppler Shift (Severe)",
        "Faulty Handover Algorithm (Too Frequent)",
        "Resource Allocation Bugs",
    ]
    for fault in expected:
        assert fault in alignment, f"Missing: {fault}"


def test_all_entries_have_required_fields(alignment):
    """Every alignment entry must have all required fields, non-empty."""
    required = [
        "3gpp_ts", "release", "clause",
        "evidence_span", "oran_component", "validated_by",
    ]
    for fault, entry in alignment.items():
        for field in required:
            assert field in entry, f"{fault} missing {field}"
            assert entry[field], f"{fault}.{field} is empty"


def test_all_3gpp_refs_valid_format(alignment):
    """All 3GPP TS references must match the TS XX.XXX format."""
    for fault, entry in alignment.items():
        ref = entry["3gpp_ts"]
        assert validate_3gpp_ref(ref), (
            f"{fault}: invalid 3GPP ref format: {ref}"
        )


def test_no_duplicate_clauses_per_ts(alignment):
    """No two faults should share the same (TS, clause) pair."""
    seen = set()
    for fault, entry in alignment.items():
        key = (entry["3gpp_ts"], entry["clause"])
        assert key not in seen, (
            f"Duplicate clause {key} found for {fault}"
        )
        seen.add(key)
