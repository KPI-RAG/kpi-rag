import json
import logging
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from src.config_loader import load_config
from src.schema import AnomalyType, ClassifierOutput, SHAPEntry
from src.utils import setup_logging, validate_3gpp_ref
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain_condition
from src.evaluator import GEvalScore

import argparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Realistic per-fault signal profiles for synthetic test payloads
# signal_statistics is flat: {"RSRP_mean": -102, "RSRP_std": 5.1, ...}
# ---------------------------------------------------------------------------
FAULT_SIGNAL_PROFILES: dict[str, dict[str, float]] = {
    "Co-Channel Interference (Mild)": {
        "RSRP_mean": -102, "RSRP_std": 5.1, "RSRP_min": -112, "RSRP_max": -91,
        "DL_SINR_mean": 4.2,  "DL_SINR_std": 2.3, "DL_SINR_min": 0.8,  "DL_SINR_max": 8.9,
        "DL_BLER_mean": 0.22, "DL_BLER_std": 0.07, "DL_BLER_min": 0.11, "DL_BLER_max": 0.35,
    },
    "Buffer Overflow (Gradual Buildup)": {
        "UL_BLER_mean": 0.44,     "UL_BLER_std": 0.11,  "UL_BLER_min": 0.29,  "UL_BLER_max": 0.63,
        "DL_PRB_UTIL_mean": 0.91, "DL_PRB_UTIL_std": 0.04, "DL_PRB_UTIL_min": 0.82, "DL_PRB_UTIL_max": 0.98,
        "DL_BLER_mean": 0.31,     "DL_BLER_std": 0.08,  "DL_BLER_min": 0.19,  "DL_BLER_max": 0.47,
    },
    "Co-Channel Interference (Severe)": {
        "RSRP_mean": -115, "RSRP_std": 6.2, "RSRP_min": -124, "RSRP_max": -103,
        "DL_SINR_mean": -1.8, "DL_SINR_std": 3.1, "DL_SINR_min": -8.2, "DL_SINR_max": 3.4,
        "DL_BLER_mean": 0.58, "DL_BLER_std": 0.12, "DL_BLER_min": 0.38, "DL_BLER_max": 0.74,
    },
    "Antenna Failure": {
        "RSRP_mean": -108, "RSRP_std": 4.2, "RSRP_min": -115, "RSRP_max": -98,
        "DL_BLER_mean": 0.38, "DL_BLER_std": 0.09, "DL_BLER_min": 0.22, "DL_BLER_max": 0.54,
        "DL_MCS_mean": 8.2,  "DL_MCS_std": 2.1, "DL_MCS_min": 4.0,  "DL_MCS_max": 13.0,
    },
    "Faulty RF Filters (Temporal)": {
        "RSRP_mean": -106, "RSRP_std": 3.8, "RSRP_min": -113, "RSRP_max": -97,
        "UL_SNR_mean": 6.1,  "UL_SNR_std": 2.9, "UL_SNR_min": 1.8,  "UL_SNR_max": 11.4,
        "DL_BLER_mean": 0.41, "DL_BLER_std": 0.10, "DL_BLER_min": 0.25, "DL_BLER_max": 0.58,
    },
    "High Network Congestion (Gradual Buildup)": {
        "DL_PRB_UTIL_mean": 0.93, "DL_PRB_UTIL_std": 0.03, "DL_PRB_UTIL_min": 0.86, "DL_PRB_UTIL_max": 0.99,
        "UL_PRB_UTIL_mean": 0.88, "UL_PRB_UTIL_std": 0.05, "UL_PRB_UTIL_min": 0.78, "UL_PRB_UTIL_max": 0.96,
        "DL_BLER_mean": 0.29, "DL_BLER_std": 0.07, "DL_BLER_min": 0.17, "DL_BLER_max": 0.42,
    },
    "Doppler Shift (Severe)": {
        "RSRP_mean": -104, "RSRP_std": 7.3, "RSRP_min": -118, "RSRP_max": -89,
        "DL_MCS_mean": 6.8,  "DL_MCS_std": 3.2, "DL_MCS_min": 2.0,  "DL_MCS_max": 14.0,
        "UL_MCS_mean": 5.9,  "UL_MCS_std": 2.8, "UL_MCS_min": 1.0,  "UL_MCS_max": 12.0,
    },
    "Faulty Handover Algorithm (Too Frequent)": {
        "RSRP_mean": -99,  "RSRP_std": 4.4, "RSRP_min": -109, "RSRP_max": -88,
        "DL_MCS_mean": 11.2, "DL_MCS_std": 2.6, "DL_MCS_min": 6.0,  "DL_MCS_max": 17.0,
        "UL_BLER_mean": 0.28, "UL_BLER_std": 0.08, "UL_BLER_min": 0.15, "UL_BLER_max": 0.41,
    },
    "Resource Allocation Bugs": {
        "DL_PRB_UTIL_mean": 0.45, "DL_PRB_UTIL_std": 0.18, "DL_PRB_UTIL_min": 0.18, "DL_PRB_UTIL_max": 0.79,
        "UL_PRB_UTIL_mean": 0.41, "UL_PRB_UTIL_std": 0.16, "UL_PRB_UTIL_min": 0.14, "UL_PRB_UTIL_max": 0.72,
        "DL_MCS_mean": 9.1,       "DL_MCS_std": 4.2,       "DL_MCS_min": 2.0,       "DL_MCS_max": 18.0,
    },
    "High Network Congestion (Sudden Spike)": {
        "DL_PRB_UTIL_mean": 0.97, "DL_PRB_UTIL_std": 0.02, "DL_PRB_UTIL_min": 0.93, "DL_PRB_UTIL_max": 1.00,
        "UL_BLER_mean": 0.51,     "UL_BLER_std": 0.13,     "UL_BLER_min": 0.31,     "UL_BLER_max": 0.69,
        "DL_BLER_mean": 0.45,     "DL_BLER_std": 0.11,     "DL_BLER_min": 0.28,     "DL_BLER_max": 0.61,
    },
}


# SHAP profiles: top 3 channels per fault with realistic shap values
FAULT_SHAP_PROFILES: dict[str, list[dict]] = {
    "Co-Channel Interference (Mild)": [
        {"channel": "DL_SINR", "shap_value": -0.38, "feature_vs_normal": "below_normal_mean"},
        {"channel": "RSRP",    "shap_value": -0.24, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_BLER", "shap_value":  0.18, "feature_vs_normal": "above_normal_mean"},
    ],
    "Buffer Overflow (Gradual Buildup)": [
        {"channel": "DL_PRB_UTIL", "shap_value":  0.45, "feature_vs_normal": "above_normal_mean"},
        {"channel": "UL_BLER",     "shap_value":  0.31, "feature_vs_normal": "above_normal_mean"},
        {"channel": "DL_BLER",     "shap_value":  0.22, "feature_vs_normal": "above_normal_mean"},
    ],
    "Co-Channel Interference (Severe)": [
        {"channel": "DL_SINR", "shap_value": -0.52, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_BLER", "shap_value":  0.41, "feature_vs_normal": "above_normal_mean"},
        {"channel": "RSRP",    "shap_value": -0.33, "feature_vs_normal": "below_normal_mean"},
    ],
    "Antenna Failure": [
        {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
        {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"},
    ],
    "Faulty RF Filters (Temporal)": [
        {"channel": "UL_SNR",  "shap_value": -0.36, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_BLER", "shap_value":  0.29, "feature_vs_normal": "above_normal_mean"},
        {"channel": "RSRP",    "shap_value": -0.21, "feature_vs_normal": "below_normal_mean"},
    ],
    "High Network Congestion (Gradual Buildup)": [
        {"channel": "DL_PRB_UTIL", "shap_value":  0.48, "feature_vs_normal": "above_normal_mean"},
        {"channel": "UL_PRB_UTIL", "shap_value":  0.35, "feature_vs_normal": "above_normal_mean"},
        {"channel": "DL_BLER",     "shap_value":  0.19, "feature_vs_normal": "above_normal_mean"},
    ],
    "Doppler Shift (Severe)": [
        {"channel": "DL_MCS",  "shap_value": -0.39, "feature_vs_normal": "below_normal_mean"},
        {"channel": "UL_MCS",  "shap_value": -0.30, "feature_vs_normal": "below_normal_mean"},
        {"channel": "RSRP",    "shap_value": -0.25, "feature_vs_normal": "below_normal_mean"},
    ],
    "Faulty Handover Algorithm (Too Frequent)": [
        {"channel": "RSRP",    "shap_value": -0.33, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_MCS",  "shap_value": -0.27, "feature_vs_normal": "below_normal_mean"},
        {"channel": "UL_BLER", "shap_value":  0.20, "feature_vs_normal": "above_normal_mean"},
    ],
    "Resource Allocation Bugs": [
        {"channel": "DL_PRB_UTIL", "shap_value": -0.40, "feature_vs_normal": "below_normal_mean"},
        {"channel": "UL_PRB_UTIL", "shap_value": -0.32, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_MCS",      "shap_value": -0.18, "feature_vs_normal": "below_normal_mean"},
    ],
    "High Network Congestion (Sudden Spike)": [
        {"channel": "DL_PRB_UTIL", "shap_value":  0.51, "feature_vs_normal": "above_normal_mean"},
        {"channel": "UL_BLER",     "shap_value":  0.38, "feature_vs_normal": "above_normal_mean"},
        {"channel": "DL_BLER",     "shap_value":  0.30, "feature_vs_normal": "above_normal_mean"},
    ],
}

# Fault types to include in Track C (Jamming excluded)
TRACK_C_FAULTS = [ft for ft in AnomalyType if ft.value != "Jamming"]


def build_synthetic_payload(fault_type: AnomalyType) -> ClassifierOutput:
    """Create a synthetic ClassifierOutput for a given fault type."""
    signals = FAULT_SIGNAL_PROFILES[fault_type.value]
    shap_entries = FAULT_SHAP_PROFILES[fault_type.value]

    return ClassifierOutput(
        anomaly_type=fault_type,
        confidence=0.85,
        shap_top3=[SHAPEntry(**s) for s in shap_entries],
        signal_statistics=signals,
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

    # Build sample set using real windows from dataset
    import json
    import random
    from src.schema import ClassifierOutput

    with open("data/processed/layer2_output_sessionsplit.json", "r") as f:
        all_windows_raw = json.load(f)

    parsed_windows = []
    for w in all_windows_raw:
        try:
            parsed_windows.append(ClassifierOutput(**w))
        except Exception:
            pass

    random.seed(cfg.get("eval", {}).get("track_c", {}).get("random_state", 42))

    by_fault = {}
    for w in parsed_windows:
        if w.anomaly_type in TRACK_C_FAULTS:
            by_fault.setdefault(w.anomaly_type, []).append(w)

    samples: list[tuple[AnomalyType, ClassifierOutput]] = []
    for ft in TRACK_C_FAULTS:
        pool = by_fault.get(ft, [])
        n = min(n_per_fault, len(pool))
        if n > 0:
            chosen = random.sample(pool, n)
            for w in chosen:
                samples.append((ft, w))

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
