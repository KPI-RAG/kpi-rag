import pytest
import json
from src.schema import LLMExplanation
from src.evaluator import (
    GEvalScore,
    TrackBResults,
    TrackCResults,
    score_explanation,
    compute_track_b,
    compute_track_c,
    load_scores_from_jsonl,
    save_results
)

def create_mock_explanation(ref):
    return LLMExplanation(
        root_cause="x",
        gpp_reference=ref,
        oran_component="x",
        recommended_action="x",
        reasoning_trace="x",
        reference_valid=True,
        template_generated=False
    )

@pytest.fixture
def synth_scores():
    scores = []
    for _ in range(3):
        scores.append(score_explanation(create_mock_explanation("TS 99.999"), "id1", 1, "Antenna Failure", 2.0, 2.0, 2.0, 2.0))
    for _ in range(3):
        scores.append(score_explanation(create_mock_explanation("TS 99.999"), "id2", 2, "Antenna Failure", 3.0, 3.0, 3.0, 3.0))
    for _ in range(3):
        scores.append(score_explanation(create_mock_explanation("TS 38.104"), "id3", 3, "Antenna Failure", 4.0, 4.0, 4.0, 4.0))
    return scores

def test_score_explanation():
    expl = create_mock_explanation("TS 38.104")
    score = score_explanation(expl, "1", 1, "Antenna Failure", 3.0, 4.0, 5.0, 2.0)
    assert isinstance(score, GEvalScore)
    assert score.overall == (3.0 + 4.0 + 5.0 + 2.0) / 4.0
    assert score.reference_valid is True
    
    expl_bad = create_mock_explanation("TS 99.999")
    score_bad = score_explanation(expl_bad, "2", 1, "Antenna Failure", 3.0, 4.0, 5.0, 2.0)
    assert score_bad.reference_valid is False
    
    with pytest.raises(ValueError):
        score_explanation(expl, "3", 1, "Antenna Failure", 6.0, 4.0, 5.0, 2.0)


def test_score_explanation_fault_specific_with_alignment():
    """With an alignment table, a well-formed but wrong-fault spec is invalid (same rule as Track C)."""
    from src.llm_explainer import load_alignment_table
    alignment = load_alignment_table("data/alignment_table.json")
    wrong = score_explanation(create_mock_explanation("TS 38.104"), "1", 3, "Antenna Failure",
                              3.0, 3.0, 3.0, 3.0, alignment=alignment)
    right = score_explanation(create_mock_explanation("TS 38.141-1"), "2", 3, "Antenna Failure",
                              3.0, 3.0, 3.0, 3.0, alignment=alignment)
    assert wrong.reference_valid is False
    assert right.reference_valid is True

def test_compute_track_b(synth_scores):
    res = compute_track_b(synth_scores)
    assert isinstance(res, TrackBResults)
    
    assert res.mean_overall == 3.0
    
    assert abs(res.citation_validity_rate - (3/9)) < 1e-6
    assert res.meets_threshold is False
    
    with pytest.raises(ValueError):
        compute_track_b([])

def test_compute_track_b_threshold(synth_scores):
    for s in synth_scores:
        s.reference_valid = True
    res = compute_track_b(synth_scores)
    assert res.citation_validity_rate == 1.0
    assert res.meets_threshold is True

def test_compute_track_c(synth_scores):
    res = compute_track_c(synth_scores)
    assert isinstance(res, TrackCResults)
    assert res.condition1_mean == 2.0
    assert res.condition2_mean == 3.0
    assert res.condition3_mean == 4.0
    assert res.condition1_citation_rate == 0.0
    assert res.condition2_citation_rate == 0.0
    assert res.condition3_citation_rate == 1.0
    
    assert res.delta_3v2 == 1.0
    assert res.delta_3v1 == 2.0
    
    with pytest.raises(ValueError):
        compute_track_c(synth_scores[:3])

def test_load_scores_from_jsonl(tmp_path, synth_scores):
    path = tmp_path / "scores.jsonl"
    with open(path, "w") as f:
        for s in synth_scores:
            d = {
                "explanation_id": s.explanation_id,
                "condition": s.condition,
                "fault_type": s.fault_type,
                "citation_validity": s.citation_validity,
                "fault_specificity": s.fault_specificity,
                "actionability": s.actionability,
                "causal_soundness": s.causal_soundness,
                "reference_valid": s.reference_valid
            }
            f.write(json.dumps(d) + "\n")
            
    loaded = load_scores_from_jsonl(str(path))
    assert len(loaded) == 9
    assert isinstance(loaded[0], GEvalScore)
    assert loaded[0].overall == synth_scores[0].overall
    
    empty_path = tmp_path / "empty.jsonl"
    empty_path.touch()
    assert load_scores_from_jsonl(str(empty_path)) == []

def test_save_results(tmp_path, synth_scores):
    path = tmp_path / "results.json"
    tb = compute_track_b(synth_scores)
    tc = compute_track_c(synth_scores)
    
    save_results(tb, tc, str(path))
    
    with open(path, "r") as f:
        data = json.load(f)
        
    assert "track_b" in data
    assert "track_c" in data
    assert data["track_b"]["n"] == 9
    assert data["track_c"]["delta_3v2"] == 1.0

def test_save_results_none_track_c(tmp_path, synth_scores):
    path = tmp_path / "results_b_only.json"
    tb = compute_track_b(synth_scores)
    
    save_results(tb, None, str(path))
    
    with open(path, "r") as f:
        data = json.load(f)
        
    assert "track_b" in data
    assert data["track_b"]["n"] == 9
    assert data["track_c"] is None



def test_geval_score_rejects_unannotated_template_row():
    """An all-0.0 template row is not a score and must not be averaged."""
    with pytest.raises(ValueError, match="outside 1-5"):
        GEvalScore(explanation_id="w1_c1", condition=1, fault_type="Antenna Failure",
                   citation_validity=0.0, fault_specificity=0.0, actionability=0.0, causal_soundness=0.0)


def test_load_scores_ignores_judge_rationale(tmp_path):
    """A G-Eval judge's extra fields (rationale) must not break loading."""
    row = {"explanation_id": "w611_c3", "condition": 3, "fault_type": "Doppler Shift (Severe)",
           "citation_validity": 4, "fault_specificity": 4, "actionability": 3, "causal_soundness": 4,
           "reference_valid": True, "rationale": "cites TR 38.901 but no clause", "overall": 9.9}
    p = tmp_path / "judged.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (s,) = load_scores_from_jsonl(str(p))
    assert s.overall == 3.75 and s.reference_valid is True
