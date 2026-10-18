import json
import logging
import random
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.config_loader import load_config
from src.schema import AnomalyType, ClassifierOutput
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain_condition, alignment_path
from src.rca_loader import RCALoader

logger = logging.getLogger(__name__)

# Jamming has no 3GPP clause, so it is not part of the citation ablation.
TRACK_C_FAULTS = [ft for ft in AnomalyType if ft.value != "Jamming"]


def sample_verified_windows(
    phase2_path: str,
    handoff_path: str,
    test_idx: set,
    fault_types: list,
    n_per_fault: int,
    seed: int = 42,
) -> dict:
    """Sample test windows where BOTH the Phase 2 prediction AND the ground truth equal the fault.

    Returns {fault_type: [layer2 records]}. Faults with fewer verified windows than
    n_per_fault get all they have (logged); nothing is topped up from other files.
    """
    rng = random.Random(seed)
    with open(phase2_path, "r", encoding="utf-8") as f:
        phase2 = json.load(f)
    with open(handoff_path, "r", encoding="utf-8") as f:
        truth = {r["window_index"]: r["ground_truth_anomaly_type"] for r in json.load(f)}

    by_fault = {}
    for ft in fault_types:
        ft_val = ft.value if hasattr(ft, "value") else ft
        verified = [
            w for w in phase2
            if w["window_index"] in test_idx
            and w.get("anomaly_type") == ft_val
            and truth.get(w["window_index"]) == ft_val   # normal windows are not in the handoff file
        ]
        if len(verified) < n_per_fault:
            logger.warning("Fault '%s': only %d verified windows (need %d), using all of them",
                           ft_val, len(verified), n_per_fault)
        by_fault[ft] = rng.sample(verified, min(n_per_fault, len(verified)))
        logger.info("Fault '%s': %d verified windows, sampled %d", ft_val, len(verified), len(by_fault[ft]))
    return by_fault


def run_track_c(
    output_dir: str,
    cfg: dict,
    n_per_fault: int = 3,
) -> None:
    """Generate explanations for all 3 conditions on verified samples.

    Up to n_per_fault windows per fault type (10 faults), each a correctly
    detected and correctly classified test window, × 3 conditions.
    Saves explanation JSONL and scores template for human annotation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    explanations_path = out / "track_c_explanations.jsonl"
    scores_path = out / "track_c_scores_template.jsonl"

    collection = get_collection(cfg)
    alignment = load_alignment_table(alignment_path(cfg))
    data_cfg = cfg.get("data", {})
    rca_loader = RCALoader(data_cfg.get("rca_evidence_path", "data/processed/rca_evidence.json"))

    processed_dir = data_cfg.get("processed_path", "data/processed")
    layer2_path = os.path.join(processed_dir, data_cfg.get("layer2_output_file", "layer2_output_sessionsplit_phase2.json"))
    handoff_path = os.path.join(processed_dir, "layer2_rag_handoff_sessionsplit.json")   # ground truth
    test_idx_path = os.path.join(
        data_cfg.get("indices_path", "data/indices"), data_cfg.get("test_idx_file", "test_idx_sessionsplit.npy")
    )
    # Held-out guard: the index is train-only, so only test windows may be explained.
    if not os.path.exists(test_idx_path):
        raise FileNotFoundError(f"Test index not found: {test_idx_path} — aborting.")
    test_set = set(np.load(test_idx_path).tolist())

    by_fault = sample_verified_windows(
        layer2_path, handoff_path, test_set, TRACK_C_FAULTS, n_per_fault,
        seed=data_cfg.get("random_state", 42),
    )
    samples = [(ft, ClassifierOutput(**w), w["window_index"]) for ft in TRACK_C_FAULTS for w in by_fault[ft]]
    logger.info("Track C: %d verified samples × 3 conditions = %d LLM calls", len(samples), len(samples) * 3)

    n_without_rca = sum(1 for ft, _, w in samples if (rca_loader.get(w) or {}).get("predicted_fault") != ft.value)
    if n_without_rca:
        logger.warning("%d/%d samples have no matching RCA record; their C3 prompt carries no RCA block",
                       n_without_rca, len(samples))

    all_explanations: list[dict] = []
    all_scores: list[dict] = []
    sample_idx = 0

    for ft, payload, window_index in samples:
        # Retrieve tickets once per sample (shared across conditions)
        tickets, _ = query_from_classifier_output(payload, collection, cfg)

        for condition in (1, 2, 3):
            sample_idx += 1
            logger.info(
                "[%d] fault=%s condition=%d window=%s ...",
                sample_idx, ft.value, condition, window_index,
            )

            # C3 gets the alignment row and RCA evidence; its citation score therefore
            # measures prompt compliance, with C1/C2 as the baselines.
            rca_context = ""
            if condition == 3 and (rca_loader.get(window_index) or {}).get("predicted_fault") == ft.value:
                rca_context = rca_loader.get_prompt_context(window_index)
                logger.info("  RCA evidence injected for window %s (%d chars)", window_index, len(rca_context))

            explanation = explain_condition(
                payload, tickets, cfg, alignment,
                condition=condition,
                rca_context=rca_context,
            )

            explanation_record = {
                "condition": condition,
                "fault_type": ft.value,
                "window_index": window_index,
                "explanation": explanation.model_dump(),
                "n_tickets": len(tickets),
            }
            all_explanations.append(explanation_record)

            # Build placeholder GEvalScore for human annotation
            score_record = {
                "explanation_id": f"w{window_index}_c{condition}",
                "condition": condition,
                "fault_type": ft.value,
                "citation_validity": 0.0,
                "fault_specificity": 0.0,
                "actionability": 0.0,
                "causal_soundness": 0.0,
                "reference_valid": explanation.reference_valid,
            }
            all_scores.append(score_record)

    # Write outputs
    with open(explanations_path, "w", encoding="utf-8") as f:
        for rec in all_explanations:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d explanations to %s", len(all_explanations), explanations_path)

    with open(scores_path, "w", encoding="utf-8") as f:
        for rec in all_scores:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d score templates to %s", len(all_scores), scores_path)

    # Compute auto-metrics (no human scores needed)
    by_condition: dict[int, list[dict]] = defaultdict(list)
    for rec in all_explanations:
        by_condition[rec["condition"]].append(rec)

    for c in (1, 2, 3):
        items = by_condition[c]
        n_total = len(items)
        n_valid = sum(
            1 for d in items if d["explanation"].get("reference_valid")
        )
        n_template = sum(
            1 for d in items if d["explanation"].get("template_generated")
        )
        logger.info(
            "Condition %d: citation_valid=%d/%d (%.0f%%)  template_fallback=%d/%d (%.0f%%)",
            c,
            n_valid, n_total, (n_valid / n_total * 100) if n_total else 0,
            n_template, n_total, (n_template / n_total * 100) if n_total else 0,
        )


def main() -> None:
    # Load .env for local development; on Streamlit Cloud env vars come from secrets.
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv()

    # Pre-load config to wire defaults
    config_path = "configs/config.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
    
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run Track C evaluation — generate explanations under 3 ablation conditions",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="directory for output files (explanations + scores template, default: data/processed)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="path to config.yaml",
    )
    parser.add_argument(
        "--n-per-fault", type=int, default=cfg.get("evaluation", {}).get("samples_per_fault", 3),
        help="samples per fault type (default: from config)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="generate 1 sample per fault instead of n-per-fault",
    )

    args = parser.parse_args()
    setup_logging(__name__)

    n = 1 if args.dry_run else args.n_per_fault

    try:
        run_track_c(args.output, cfg, n_per_fault=n)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Value error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
