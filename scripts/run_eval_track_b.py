from src.config_loader import load_config
from src.utils import setup_logging
from src.evaluator import (
    load_scores_from_jsonl,
    compute_track_b,
    save_results
)
import argparse, json, logging, sys

logger = logging.getLogger(__name__)

def run_track_b(scores_path: str, output_path: str, cfg: dict) -> None:
    scores = load_scores_from_jsonl(scores_path)
    results = compute_track_b(scores)
    
    print(f"Track B Results — n={results.n}")
    print(f"  Citation Validity Rate : {results.citation_validity_rate:.1%}")
    print(f"  Mean Overall Score     : {results.mean_overall:.2f}/5.0")
    print(f"  Meets 70% threshold   : {results.meets_threshold}")
    
    save_results(results, None, output_path)
    logger.info("Track B results saved to %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track B evaluation")
    parser.add_argument("--scores", type=str, required=True, help="path to JSONL file of GEvalScores")
    parser.add_argument("--output", type=str, required=True, help="path to write results JSON")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path to config.yaml")
    
    args = parser.parse_args()
    
    setup_logging(__name__)
    
    try:
        cfg = load_config(args.config)
        run_track_b(args.scores, args.output, cfg)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Value error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
