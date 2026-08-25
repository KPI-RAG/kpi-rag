import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.config_loader import load_config
from src.schema import AnomalyType, ClassifierOutput, SignalStats, SHAPEntry
from src.utils import setup_logging, validate_3gpp_ref
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain_condition
from src.evaluator import GEvalScore

import argparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Realistic per-fault signal profiles for synthetic test payloads
# ---------------------------------------------------------------------------
FAULT_SIGNAL_PROFILES: dict[str, dict[str, dict]] = {
    "Co-Channel Interference (Mild)": {
        "RSRP":    {"mean": -102, "std": 5.1, "min": -112, "max": -91},
        "DL_SINR": {"mean": 4.2,  "std": 2.3, "min": 0.8,  "max": 8.9},
        "DL_BLER": {"mean": 0.22, "std": 0.07, "min": 0.11, "max": 0.35},
    },
    "Buffer Overflow (Gradual Buildup)": {
        "UL_BLER":     {"mean": 0.44, "std": 0.11, "min": 0.29, "max": 0.63},
        "DL_PRB_UTIL": {"mean": 0.91, "std": 0.04, "min": 0.82, "max": 0.98},
        "DL_BLER":     {"mean": 0.31, "std": 0.08, "min": 0.19, "max": 0.47},
    },
    "Co-Channel Interference (Severe)": {
        "RSRP":    {"mean": -115, "std": 6.2, "min": -124, "max": -103},
        "DL_SINR": {"mean": -1.8, "std": 3.1, "min": -8.2, "max": 3.4},
        "DL_BLER": {"mean": 0.58, "std": 0.12, "min": 0.38, "max": 0.74},
    },
    "Antenna Failure": {
        "RSRP":    {"mean": -108, "std": 4.2, "min": -115, "max": -98},
        "DL_BLER": {"mean": 0.38, "std": 0.09, "min": 0.22, "max": 0.54},
        "DL_MCS":  {"mean": 8.2,  "std": 2.1, "min": 4.0,  "max": 13.0},
    },
    "Faulty RF Filters (Temporal)": {
        "RSRP":    {"mean": -106, "std": 3.8, "min": -113, "max": -97},
        "UL_SNR":  {"mean": 6.1,  "std": 2.9, "min": 1.8,  "max": 11.4},
        "DL_BLER": {"mean": 0.41, "std": 0.10, "min": 0.25, "max": 0.58},
    },
    "High Network Congestion (Gradual Buildup)": {
        "DL_PRB_UTIL": {"mean": 0.93, "std": 0.03, "min": 0.86, "max": 0.99},
        "UL_PRB_UTIL": {"mean": 0.88, "std": 0.05, "min": 0.78, "max": 0.96},
        "DL_BLER":     {"mean": 0.29, "std": 0.07, "min": 0.17, "max": 0.42},
    },
    "Doppler Shift (Severe)": {
        "RSRP":   {"mean": -104, "std": 7.3, "min": -118, "max": -89},
        "DL_MCS": {"mean": 6.8,  "std": 3.2, "min": 2.0,  "max": 14.0},
        "UL_MCS": {"mean": 5.9,  "std": 2.8, "min": 1.0,  "max": 12.0},
    },
    "Faulty Handover Algorithm (Too Frequent)": {
        "RSRP":    {"mean": -99,  "std": 4.4, "min": -109, "max": -88},
        "DL_MCS":  {"mean": 11.2, "std": 2.6, "min": 6.0,  "max": 17.0},
        "UL_BLER": {"mean": 0.28, "std": 0.08, "min": 0.15, "max": 0.41},
    },
    "Resource Allocation Bugs": {
        "DL_PRB_UTIL": {"mean": 0.45, "std": 0.18, "min": 0.18, "max": 0.79},
        "UL_PRB_UTIL": {"mean": 0.41, "std": 0.16, "min": 0.14, "max": 0.72},
        "DL_MCS":      {"mean": 9.1,  "std": 4.2,  "min": 2.0,  "max": 18.0},
    },
    "High Network Congestion (Sudden Spike)": {
        "DL_PRB_UTIL": {"mean": 0.97, "std": 0.02, "min": 0.93, "max": 1.00},
        "UL_BLER":     {"mean": 0.51, "std": 0.13, "min": 0.31, "max": 0.69},
        "DL_BLER":     {"mean": 0.45, "std": 0.11, "min": 0.28, "max": 0.61},
    },
}

# SHAP profiles: top 3 channels per fault with realistic shap values
FAULT_SHAP_PROFILES: dict[str, list[dict]] = {
    "Co-Channel Interference (Mild)": [
        {"channel": "DL_SINR", "shap_value": -0.38, "direction": "below_normal"},
        {"channel": "RSRP",    "shap_value": -0.24, "direction": "below_normal"},
        {"channel": "DL_BLER", "shap_value":  0.18, "direction": "above_normal"},
    ],
    "Buffer Overflow (Gradual Buildup)": [
        {"channel": "DL_PRB_UTIL", "shap_value":  0.45, "direction": "above_normal"},
        {"channel": "UL_BLER",     "shap_value":  0.31, "direction": "above_normal"},
        {"channel": "DL_BLER",     "shap_value":  0.22, "direction": "above_normal"},
    ],
    "Co-Channel Interference (Severe)": [
        {"channel": "DL_SINR", "shap_value": -0.52, "direction": "below_normal"},
        {"channel": "DL_BLER", "shap_value":  0.41, "direction": "above_normal"},
        {"channel": "RSRP",    "shap_value": -0.33, "direction": "below_normal"},
    ],
    "Antenna Failure": [
        {"channel": "RSRP",    "shap_value": -0.42, "direction": "below_normal"},
        {"channel": "DL_BLER", "shap_value":  0.28, "direction": "above_normal"},
        {"channel": "DL_MCS",  "shap_value": -0.19, "direction": "below_normal"},
    ],
    "Faulty RF Filters (Temporal)": [
        {"channel": "UL_SNR",  "shap_value": -0.36, "direction": "below_normal"},
        {"channel": "DL_BLER", "shap_value":  0.29, "direction": "above_normal"},
        {"channel": "RSRP",    "shap_value": -0.21, "direction": "below_normal"},
    ],
    "High Network Congestion (Gradual Buildup)": [
        {"channel": "DL_PRB_UTIL", "shap_value":  0.48, "direction": "above_normal"},
        {"channel": "UL_PRB_UTIL", "shap_value":  0.35, "direction": "above_normal"},
        {"channel": "DL_BLER",     "shap_value":  0.19, "direction": "above_normal"},
    ],
    "Doppler Shift (Severe)": [
        {"channel": "DL_MCS",  "shap_value": -0.39, "direction": "below_normal"},
        {"channel": "UL_MCS",  "shap_value": -0.30, "direction": "below_normal"},
        {"channel": "RSRP",    "shap_value": -0.25, "direction": "below_normal"},
    ],
    "Faulty Handover Algorithm (Too Frequent)": [
        {"channel": "RSRP",    "shap_value": -0.33, "direction": "below_normal"},
        {"channel": "DL_MCS",  "shap_value": -0.27, "direction": "below_normal"},
        {"channel": "UL_BLER", "shap_value":  0.20, "direction": "above_normal"},
    ],
    "Resource Allocation Bugs": [
        {"channel": "DL_PRB_UTIL", "shap_value": -0.40, "direction": "below_normal"},
        {"channel": "UL_PRB_UTIL", "shap_value": -0.32, "direction": "below_normal"},
        {"channel": "DL_MCS",      "shap_value": -0.18, "direction": "below_normal"},
    ],
    "High Network Congestion (Sudden Spike)": [
        {"channel": "DL_PRB_UTIL", "shap_value":  0.51, "direction": "above_normal"},
        {"channel": "UL_BLER",     "shap_value":  0.38, "direction": "above_normal"},
        {"channel": "DL_BLER",     "shap_value":  0.30, "direction": "above_normal"},
    ],
}

# Fault types to include in Track C (Jamming excluded)
TRACK_C_FAULTS = [ft for ft in AnomalyType if ft.value != "Jamming"]


def build_synthetic_payload(fault_type: AnomalyType) -> ClassifierOutput:
    """Create a synthetic ClassifierOutput for a given fault type."""
    signals = FAULT_SIGNAL_PROFILES[fault_type.value]
    shap_entries = FAULT_SHAP_PROFILES[fault_type.value]

    signal_statistics = {
        k: SignalStats(**v)
        for k, v in signals.items()
    }

    return ClassifierOutput(
        anomaly_type=fault_type,
        confidence=0.85,
        shap_top3=[SHAPEntry(**s) for s in shap_entries],
        signal_statistics=signal_statistics,
    )


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

    # Build sample set: n_per_fault identical payloads per fault type
    samples: list[tuple[AnomalyType, ClassifierOutput]] = []
    for ft in TRACK_C_FAULTS:
        for _ in range(n_per_fault):
            samples.append((ft, build_synthetic_payload(ft)))

    logger.info(
        "Track C: %d fault types × %d samples × 3 conditions = %d LLM calls",
        len(TRACK_C_FAULTS), n_per_fault, len(samples) * 3,
    )

    all_explanations: list[dict] = []
    all_scores: list[dict] = []
    sample_idx = 0

    for ft, payload in samples:
        # Retrieve tickets once per sample (shared across conditions)
        tickets, _ = query_from_classifier_output(payload, collection, cfg)

        for condition in (1, 2, 3):
            sample_idx += 1
            logger.info(
                "[%d] fault=%s condition=%d ...",
                sample_idx, ft.value, condition,
            )

            explanation = explain_condition(
                payload, tickets, cfg, alignment, condition=condition,
            )

            explanation_record = {
                "condition": condition,
                "fault_type": ft.value,
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
        "--n-per-fault", type=int, default=3,
        help="samples per fault type (default: 3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="generate 1 sample per fault instead of n-per-fault",
    )

    args = parser.parse_args()
    setup_logging(__name__)

    n = 1 if args.dry_run else args.n_per_fault

    try:
        cfg = load_config(args.config)
        run_track_c(args.output, cfg, n_per_fault=n)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Value error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
