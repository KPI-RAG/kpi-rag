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

def compute_track_b(scores: list[GEvalScore]) -> TrackBResults:
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
    meets_threshold = citation_validity_rate >= 0.70
    
    logger.info("Track B computed for n=%d samples", n)
    logger.info("Citation validity rate: %.3f (meets_threshold=%s)", citation_validity_rate, meets_threshold)
    
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
    track_b: TrackBResults,
    track_c: TrackCResults,
    path: str
) -> None:
    output = {
        "track_b": asdict(track_b),
        "track_c": asdict(track_c)
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
