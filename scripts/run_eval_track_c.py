from src.config_loader import load_config
from src.utils import setup_logging
from src.evaluator import (
    load_scores_from_jsonl,
    compute_track_b,
    compute_track_c,
    save_results
)
import argparse, logging, sys

logger = logging.getLogger(__name__)

def run_track_c(scores_path: str, output_path: str, cfg: dict) -> None:
    scores = load_scores_from_jsonl(scores_path)
    track_b = compute_track_b(scores)
    track_c = compute_track_c(scores)
    
    print("Track C Ablation Results")
    print(f"  Condition 1 (label only)       : {track_c.condition1_mean:.2f} | citation rate: {track_c.condition1_citation_rate:.1%}")
    print(f"  Condition 2 (+ tickets)        : {track_c.condition2_mean:.2f} | citation rate: {track_c.condition2_citation_rate:.1%}")
    print(f"  Condition 3 (full system)      : {track_c.condition3_mean:.2f} | citation rate: {track_c.condition3_citation_rate:.1%}")
    print(f"  Delta 3v2 (grounding effect)   : {track_c.delta_3v2:+.2f}")
    print(f"  Delta 3v1 (full improvement)   : {track_c.delta_3v1:+.2f}")
    
    save_results(track_b, track_c, output_path)
    logger.info("Track C results saved to %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track C evaluation")
    parser.add_argument("--scores", type=str, required=True, help="path to JSONL file (all 3 conditions)")
    parser.add_argument("--output", type=str, required=True, help="path to write results JSON")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path to config.yaml")
    
    args = parser.parse_args()
    setup_logging(__name__)
    
    try:
        cfg = load_config(args.config)
        run_track_c(args.scores, args.output, cfg)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Value error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
