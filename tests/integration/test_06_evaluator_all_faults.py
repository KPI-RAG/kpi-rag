"""Test 06 — Evaluator for all 10 fault types (Jamming excluded)."""
import pytest
from src.schema import LLMExplanation
from src.evaluator import (
    score_explanation,
    compute_track_b,
    compute_track_c,
    GEvalScore,
)


NON_JAMMING_FAULTS = [
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


@pytest.fixture
def mock_explanation():
    """A valid LLMExplanation with a recognized 3GPP reference."""
    return LLMExplanation(
        root_cause="Test root cause",
        gpp_reference="TS 38.104",
        oran_component="WG4 Open Fronthaul",
        recommended_action="Test action",
        reasoning_trace="Test trace",
        reference_valid=True,
        template_generated=False,
    )


def test_score_explanation_all_fault_types(mock_explanation):
    """score_explanation must work for all 10 non-Jamming fault types."""
    for fault in NON_JAMMING_FAULTS:
        score = score_explanation(
            explanation=mock_explanation,
            explanation_id=f"test_{fault}",
            condition=3,
            fault_type=fault,
            citation_validity=4.0,
            fault_specificity=4.0,
            actionability=3.5,
            causal_soundness=4.0,
        )
        assert score.overall == pytest.approx(3.875)
        assert score.fault_type == fault
        assert score.condition == 3


def test_compute_track_b_all_conditions(mock_explanation):
    """Track B with 30 synthetic scores (3 per fault × 10 faults)."""
    scores = []
    for fault in NON_JAMMING_FAULTS:
        for i in range(3):
            scores.append(
                score_explanation(
                    explanation=mock_explanation,
                    explanation_id=f"{fault}_{i}",
                    condition=3,
                    fault_type=fault,
                    citation_validity=4.0,
                    fault_specificity=4.0,
                    actionability=3.5,
                    causal_soundness=4.0,
                )
            )

    results = compute_track_b(scores)
    assert results.n == 30
    assert results.citation_validity_rate == 1.0
    assert results.meets_threshold is True


def test_compute_track_c_three_conditions():
    """Track C with 9 scores: 3 per condition, different quality levels."""
    scores = []

    # Condition 1: low quality, no valid references
    for i in range(3):
        scores.append(GEvalScore(
            explanation_id=f"c1_{i}",
            condition=1,
            fault_type="Antenna Failure",
            citation_validity=2.0,
            fault_specificity=2.0,
            actionability=2.0,
            causal_soundness=2.0,
            reference_valid=False,
        ))

    # Condition 2: medium quality, no valid references
    for i in range(3):
        scores.append(GEvalScore(
            explanation_id=f"c2_{i}",
            condition=2,
            fault_type="Antenna Failure",
            citation_validity=3.0,
            fault_specificity=3.0,
            actionability=3.0,
            causal_soundness=3.0,
            reference_valid=False,
        ))

    # Condition 3: high quality, valid references
    for i in range(3):
        scores.append(GEvalScore(
            explanation_id=f"c3_{i}",
            condition=3,
            fault_type="Antenna Failure",
            citation_validity=4.0,
            fault_specificity=4.0,
            actionability=4.0,
            causal_soundness=4.0,
            reference_valid=True,
        ))

    results = compute_track_c(scores)
    assert results.delta_3v2 == pytest.approx(1.0)
    assert results.delta_3v1 == pytest.approx(2.0)
    assert results.condition3_citation_rate == 1.0
    assert results.condition1_citation_rate == 0.0


def test_citation_validity_threshold(mock_explanation):
    """70% threshold: 21/30 valid → meets, 20/30 → does not."""
    # 21/30 valid (70% exactly)
    scores_pass = []
    for i in range(30):
        exp = LLMExplanation(
            root_cause="Test",
            gpp_reference="TS 38.104" if i < 21 else "TS 99.999",
            oran_component="WG4",
            recommended_action="Test",
            reasoning_trace="Test",
            reference_valid=i < 21,
            template_generated=False,
        )
        scores_pass.append(
            score_explanation(
                explanation=exp,
                explanation_id=f"pass_{i}",
                condition=3,
                fault_type="Antenna Failure",
                citation_validity=4.0,
                fault_specificity=4.0,
                actionability=3.5,
                causal_soundness=4.0,
            )
        )

    results_pass = compute_track_b(scores_pass)
    assert results_pass.meets_threshold is True  # >= 0.70

    # 20/30 valid (66.7%)
    scores_fail = []
    for i in range(30):
        exp = LLMExplanation(
            root_cause="Test",
            gpp_reference="TS 38.104" if i < 20 else "TS 99.999",
            oran_component="WG4",
            recommended_action="Test",
            reasoning_trace="Test",
            reference_valid=i < 20,
            template_generated=False,
        )
        scores_fail.append(
            score_explanation(
                explanation=exp,
                explanation_id=f"fail_{i}",
                condition=3,
                fault_type="Antenna Failure",
                citation_validity=4.0,
                fault_specificity=4.0,
                actionability=3.5,
                causal_soundness=4.0,
            )
        )

    results_fail = compute_track_b(scores_fail)
    assert results_fail.meets_threshold is False  # < 0.70
