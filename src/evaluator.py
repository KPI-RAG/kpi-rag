import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from src.schema import LLMExplanation
from src.utils import validate_3gpp_ref

logger = logging.getLogger(__name__)

@dataclass
class GEvalScore:
    explanation_id: str
    condition: int
    fault_type: str
    citation_validity: float
    fault_specificity: float
    actionability: float
    causal_soundness: float
    reference_valid: bool = False
    overall: float = field(init=False)

    def __post_init__(self):
        self.overall = (self.citation_validity + self.fault_specificity + 
                        self.actionability + self.causal_soundness) / 4.0

@dataclass
class TrackBResults:
    scores: list[GEvalScore]
    mean_citation_validity: float
    mean_fault_specificity: float
    mean_actionability: float
    mean_causal_soundness: float
    mean_overall: float
    citation_validity_rate: float
    n: int
    meets_threshold: bool

@dataclass
class TrackCResults:
    condition1_mean: float
    condition2_mean: float
    condition3_mean: float
    condition1_citation_rate: float
    condition2_citation_rate: float
    condition3_citation_rate: float
    delta_2v1: float
    delta_3v2: float
    delta_3v1: float

def score_explanation(
    explanation: LLMExplanation,
    explanation_id: str,
    condition: int,
    fault_type: str,
    citation_validity: float,
    fault_specificity: float,
    actionability: float,
    causal_soundness: float
) -> GEvalScore:
    """
    Validate G-Eval dimension scores and construct a ``GEvalScore`` record.

    Validates that every numeric score falls within the 1–5 scale, attempts to
    verify the 3GPP reference string embedded in the explanation, and returns a
    fully-populated :class:`GEvalScore` dataclass whose ``overall`` field is
    computed automatically in ``__post_init__``.

    Parameters
    ----------
    explanation : LLMExplanation
        The LLM-generated explanation object whose ``gpp_reference`` field is
        checked for validity against the 3GPP reference format.
    explanation_id : str
        Unique identifier for the explanation being scored (e.g. a UUID or
        dataset row key).
    condition : int
        Experimental condition under which the explanation was generated
        (1 = baseline, 2 = RAG, 3 = RAG + standards grounding).
    fault_type : str
        Category label for the network fault described in the explanation.
    citation_validity : float
        G-Eval score (1–5) measuring whether the cited 3GPP reference is
        correctly identified and relevant.
    fault_specificity : float
        G-Eval score (1–5) measuring how precisely the explanation identifies
        the specific fault rather than giving a generic answer.
    actionability : float
        G-Eval score (1–5) measuring whether the explanation provides concrete,
        actionable remediation steps.
    causal_soundness : float
        G-Eval score (1–5) measuring the logical correctness of the causal
        chain presented in the explanation.

    Returns
    -------
    GEvalScore
        A dataclass instance populated with all provided scores plus a
        ``reference_valid`` flag derived from 3GPP reference validation and an
        auto-computed ``overall`` mean across the four dimensions.

    Raises
    ------
    ValueError
        If any of the four numeric scores is outside the closed interval [1, 5].
    """
    for val in [citation_validity, fault_specificity, actionability, causal_soundness]:
        if not (1.0 <= val <= 5.0):
            raise ValueError(f"Score {val} outside 1-5 range")
            
    try:
        ref_valid = bool(validate_3gpp_ref(explanation.gpp_reference))
    except Exception:
        ref_valid = False
        
    return GEvalScore(
        explanation_id=explanation_id,
        condition=condition,
        fault_type=fault_type,
        citation_validity=citation_validity,
        fault_specificity=fault_specificity,
        actionability=actionability,
        causal_soundness=causal_soundness,
        reference_valid=ref_valid
    )

def compute_track_b(scores: list[GEvalScore], cfg: dict | None = None) -> TrackBResults:
    """Compute overall metrics for Track B (Baseline).
    
    Parameters
    ----------
    scores : list[GEvalScore]
        List of G-Eval scores for Track B explanations.
    cfg : dict | None, optional
        Application configuration. Used to read the threshold for
        citation validity.
        
    Returns
    -------
    TrackBResults
        The aggregated metrics for Track B.
    """
    if not scores:
        raise ValueError("Scores list is empty")
        
    n = len(scores)
    mean_citation_validity = statistics.mean([s.citation_validity for s in scores])
    mean_fault_specificity = statistics.mean([s.fault_specificity for s in scores])
    mean_actionability = statistics.mean([s.actionability for s in scores])
    mean_causal_soundness = statistics.mean([s.causal_soundness for s in scores])
    mean_overall = statistics.mean([s.overall for s in scores])
    
    valid_count = sum(1 for s in scores if s.reference_valid)
    citation_validity_rate = valid_count / n
    # Read threshold from config; fall back to 0.70 if cfg not provided.
    threshold = (
        cfg.get("evaluation", {}).get("citation_validity_threshold", 0.70)
        if cfg is not None
        else 0.70
    )
    meets_threshold = citation_validity_rate >= threshold
    
    logger.info("Track B computed for n=%d samples", n)
    logger.info("Citation validity rate: %.3f (threshold=%.2f, meets_threshold=%s)",
                citation_validity_rate, threshold, meets_threshold)
    
    return TrackBResults(
        scores=scores,
        mean_citation_validity=mean_citation_validity,
        mean_fault_specificity=mean_fault_specificity,
        mean_actionability=mean_actionability,
        mean_causal_soundness=mean_causal_soundness,
        mean_overall=mean_overall,
        citation_validity_rate=citation_validity_rate,
        n=n,
        meets_threshold=meets_threshold
    )

def compute_track_c(scores: list[GEvalScore]) -> TrackCResults:
    """Compute overall metrics for Track C (Ablation).
    
    Parameters
    ----------
    scores : list[GEvalScore]
        List of G-Eval scores across all three ablation conditions.
        
    Returns
    -------
    TrackCResults
        The aggregated ablation study metrics.
    """
    cond1 = [s for s in scores if s.condition == 1]
    cond2 = [s for s in scores if s.condition == 2]
    cond3 = [s for s in scores if s.condition == 3]
    
    if not cond1 or not cond2 or not cond3:
        raise ValueError("Missing scores for one or more conditions")
        
    cond1_mean = statistics.mean([s.overall for s in cond1])
    cond2_mean = statistics.mean([s.overall for s in cond2])
    cond3_mean = statistics.mean([s.overall for s in cond3])
    
    cond1_cit_rate = sum(1 for s in cond1 if s.reference_valid) / len(cond1)
    cond2_cit_rate = sum(1 for s in cond2 if s.reference_valid) / len(cond2)
    cond3_cit_rate = sum(1 for s in cond3 if s.reference_valid) / len(cond3)
    
    delta_2v1 = cond2_mean - cond1_mean
    delta_3v2 = cond3_mean - cond2_mean
    delta_3v1 = cond3_mean - cond1_mean
    
    logger.info("delta_3v2 (standards-grounding contribution): %.3f", delta_3v2)
    
    return TrackCResults(
        condition1_mean=cond1_mean,
        condition2_mean=cond2_mean,
        condition3_mean=cond3_mean,
        condition1_citation_rate=cond1_cit_rate,
        condition2_citation_rate=cond2_cit_rate,
        condition3_citation_rate=cond3_cit_rate,
        delta_2v1=delta_2v1,
        delta_3v2=delta_3v2,
        delta_3v1=delta_3v1
    )

def load_scores_from_jsonl(path: str) -> list[GEvalScore]:
    """Load GEvalScore records from a JSON Lines file.
    
    Parameters
    ----------
    path : str
        Path to the JSONL file containing the scores.
        
    Returns
    -------
    list[GEvalScore]
        List of parsed G-Eval score dataclasses.
    """
    scores = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "overall" in data:
                del data["overall"]
            scores.append(GEvalScore(**data))
            
    logger.info("Loaded %d GEvalScores from %s", len(scores), path)
    return scores

def save_results(
    track_b: TrackBResults | None,
    track_c: TrackCResults | None,
    path: str
) -> None:
    """Save the aggregated evaluation results to a JSON file.
    
    Parameters
    ----------
    track_b : TrackBResults | None
        Results from Track B, or None if skipped.
    track_c : TrackCResults | None
        Results from Track C, or None if skipped.
    path : str
        Output file path for the results JSON.
    """
    output = {
        "track_b": asdict(track_b) if track_b is not None else None,
        "track_c": asdict(track_c) if track_c is not None else None
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
