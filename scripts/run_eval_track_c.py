import json
import logging
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_config
from src.schema import AnomalyType
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain_condition
from src.rca_loader import RCALoader

logger = logging.getLogger(__name__)

# Jamming has no 3GPP clause, so it is not part of the citation ablation.
TRACK_C_FAULTS = [ft for ft in AnomalyType if ft.value != "Jamming"]


def run_track_c(
    output_dir: str,
    cfg: dict,
    n_per_fault: int = 3,
) -> None:
    """Generate explanations for all 3 conditions on stratified samples.

    n_per_fault samples per fault type (10 faults) × 3 conditions
    = 30 × 3 = 90 total LLM calls (or 10 × 3 = 30 in dry-run).
    Saves explanation JSONL and scores template for human annotation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    explanations_path = out / "track_c_explanations.jsonl"
    scores_path = out / "track_c_scores_template.jsonl"

    collection = get_collection(cfg)
    alignment = load_alignment_table("configs/alignment_table.json")

    # Initialize RCALoader once — O(1) lookup per window during the eval loop
    rca_evidence_path = cfg.get("data", {}).get(
        "rca_evidence_path", "data/processed/rca_evidence.json"
    )
    rca_loader = RCALoader(rca_evidence_path)

    import random
    from src.schema import ClassifierOutput

    with open("data/processed/layer2_output_sessionsplit.json", "r") as f:
        all_windows_raw = json.load(f)

    parsed_windows = []
    for w in all_windows_raw:
        try:
            payload_obj = ClassifierOutput(**w)
            window_idx = w.get("window_index")  # carry the join key for RCALoader
            parsed_windows.append((payload_obj, window_idx))
        except Exception:
            pass

    random.seed(cfg.get("data", {}).get("random_state", 42))

    by_fault: dict[AnomalyType, list[tuple[ClassifierOutput, int | None]]] = {}
    for payload_obj, window_idx in parsed_windows:
        if payload_obj.anomaly_type in TRACK_C_FAULTS:
            by_fault.setdefault(payload_obj.anomaly_type, []).append(
                (payload_obj, window_idx)
            )

    # samples: list of (fault_type, ClassifierOutput, window_index)
    samples: list[tuple[AnomalyType, ClassifierOutput, int | None]] = []
    for ft in TRACK_C_FAULTS:
        pool = by_fault.get(ft, [])
        n = min(n_per_fault, len(pool))
        if n > 0:
            chosen = random.sample(pool, n)
            for payload_obj, window_idx in chosen:
                samples.append((ft, payload_obj, window_idx))

    logger.info(
        "Track C: %d fault types × %d samples × 3 conditions = %d LLM calls",
        len(TRACK_C_FAULTS), n_per_fault, len(samples) * 3,
    )

    # Held-out guard: every sampled window must be in the test split (the index is train-only).
    data_cfg = cfg.get("data", {})
    test_idx_path = os.path.join(
        data_cfg.get("indices_path", "data/indices"), data_cfg.get("test_idx_file", "test_idx_sessionsplit.npy")
    )
    if os.path.exists(test_idx_path):
        import numpy as np
        test_set = set(np.load(test_idx_path).tolist())
        leaked = [w for _, _, w in samples if w is not None and w not in test_set]
        if leaked:
            raise ValueError(f"Track C sampled {len(leaked)} windows outside the test split: {leaked[:5]}")
    n_without_rca = sum(1 for _, _, w in samples if w is None or rca_loader.get(w) is None)
    if n_without_rca:
        logger.warning(
            "%d/%d sampled windows have no RCA record (window_index >= 1235 = normal windows "
            "that Layer 2 flagged as faults); their C3 prompt will carry no RCA evidence",
            n_without_rca, len(samples),
        )

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
            if condition == 3 and window_index is not None:
                rca_context = rca_loader.get_prompt_context(window_index)
                if rca_context:
                    logger.info(
                        "  RCA evidence injected for window %s (%d chars)",
                        window_index, len(rca_context),
                    )

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
                "explanation_id": f"{ft.value}_c{condition}_{sample_idx}",
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
        "--output", type=str, required=True,
        help="directory for output files (explanations + scores template)",
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
