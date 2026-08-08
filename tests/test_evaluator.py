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
