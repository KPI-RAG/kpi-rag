"""Track C — citation ablation (C1 label+SHAP, C2 + retrieved incidents, C3 + alignment row + RCA).

Sampling is by GROUND TRUTH, and every sample goes through what the system actually
does (Phase 2 prediction -> retrieval -> LLM). Each citation is scored twice:
  citation_correct  — against the TRUE fault's alignment row (end-to-end correctness)
  reference_valid   — against the PREDICTED fault's row (what the validator reports;
                      for C3 this is prompt compliance, since that row is in the prompt)
A window the detector misses gets no explanation and counts as not correct.
"""
import json
import logging
import random
import sys
import os
import argparse
from pathlib import Path

import numpy as np

from src.config_loader import load_config
from src.schema import AnomalyType, ClassifierOutput
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain_condition, alignment_path, validate_citation
from src.rca_loader import RCALoader

logger = logging.getLogger(__name__)

# Jamming has no 3GPP clause, so it is not part of the citation ablation.
TRACK_C_FAULTS = [ft for ft in AnomalyType if ft.value != "Jamming"]
CONDITIONS = (1, 2, 3)


def sample_by_ground_truth(
    phase2_path: str,
    handoff_path: str,
    test_idx: set,
    fault_types: list,
    n_per_fault: int,
    seed: int = 42,
) -> list[tuple[str, dict]]:
    """n_per_fault test windows per TRUE fault type, with their Phase 2 records.

    Returns [(true_fault, phase2_record)]; the record's anomaly_type is whatever the
    system predicted (possibly "Normal" or another fault).
    """
    rng = random.Random(seed)
    with open(phase2_path, "r", encoding="utf-8") as f:
        phase2 = {w["window_index"]: w for w in json.load(f)}
    with open(handoff_path, "r", encoding="utf-8") as f:
        truth = {r["window_index"]: r["ground_truth_anomaly_type"] for r in json.load(f)}

    samples = []
    for ft in fault_types:
        ft_val = ft.value if hasattr(ft, "value") else ft
        pool = sorted(w for w, t in truth.items() if t == ft_val and w in test_idx and w in phase2)
        if len(pool) < n_per_fault:
            logger.warning("True fault '%s': only %d test windows (need %d)", ft_val, len(pool), n_per_fault)
        chosen = rng.sample(pool, min(n_per_fault, len(pool)))
        samples += [(ft_val, phase2[w]) for w in chosen]
        logger.info("True fault '%s': %d test windows, sampled %s (predicted: %s)", ft_val, len(pool), chosen,
                    [phase2[w]["anomaly_type"] for w in chosen])
    return samples


def summarise(records: list[dict]) -> dict:
    """Per-condition counts: end-to-end correctness, the correctly-classified subset, compliance."""
    out = {}
    for c in CONDITIONS:
        rows = [r for r in records if r["condition"] == c]
        explained = [r for r in rows if r["explanation"] is not None]
        classified = [r for r in rows if r["predicted_fault"] == r["true_fault"]]
        out[f"C{c}"] = {
            "n": len(rows),
            "citation_correct": sum(r["citation_correct"] for r in rows),
            "correctly_classified": len(classified),
            "citation_correct_when_classified": sum(r["citation_correct"] for r in classified),
            "explained": len(explained),
            "reference_valid_vs_predicted": sum(bool(r["reference_valid"]) for r in explained),
            "missed_by_detector": sum(r["predicted_fault"] == "Normal" for r in rows),
            "template_fallbacks": sum(r["explanation"]["template_generated"] for r in explained),
        }
    return out


def run_track_c(
    output_dir: str,
    cfg: dict,
    n_per_fault: int = 3,
) -> dict:
    """Explain n_per_fault ground-truth windows per fault under C1/C2/C3; write JSONL + summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    samples = sample_by_ground_truth(
        layer2_path, handoff_path, test_set, TRACK_C_FAULTS, n_per_fault,
        seed=data_cfg.get("random_state", 42),
    )
    n_calls = sum(rec["anomaly_type"] != "Normal" for _, rec in samples) * len(CONDITIONS)
    logger.info("Track C: %d samples, %d LLM calls", len(samples), n_calls)

    records: list[dict] = []
    for true_fault, rec in samples:
        window_index, predicted = rec["window_index"], rec["anomaly_type"]
        if predicted == "Normal":   # detector miss: the system produces no explanation
            records += [{"condition": c, "window_index": window_index, "true_fault": true_fault,
                         "predicted_fault": predicted, "explanation": None, "reference_valid": None,
                         "citation_correct": False, "n_tickets": 0} for c in CONDITIONS]
            continue

        payload = ClassifierOutput(**rec)
        tickets, _ = query_from_classifier_output(payload, collection, cfg)   # shared by C2/C3
        rca_record = rca_loader.get(window_index) or {}
        for condition in CONDITIONS:
            logger.info("window=%s true=%s predicted=%s condition=%d", window_index, true_fault, predicted, condition)
            rca_context = ""
            if condition == 3 and rca_record.get("predicted_fault") == predicted:
                rca_context = rca_loader.get_prompt_context(window_index)
            explanation = explain_condition(payload, tickets, cfg, alignment, condition=condition, rca_context=rca_context)
            records.append({
                "condition": condition,
                "window_index": window_index,
                "true_fault": true_fault,
                "predicted_fault": predicted,
                "explanation": explanation.model_dump(),
                "reference_valid": explanation.reference_valid,
                "citation_correct": (not explanation.template_generated)
                                    and validate_citation(explanation.gpp_reference, alignment, true_fault),
                "n_tickets": len(tickets),
                "rca_injected": bool(rca_context),
            })

    with open(out / "track_c_explanations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Human G-Eval template: one row per generated explanation, rated against the TRUE fault
    with open(out / "track_c_scores_template.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            if r["explanation"] is not None:
                f.write(json.dumps({
                    "explanation_id": f"w{r['window_index']}_c{r['condition']}",
                    "condition": r["condition"],
                    "fault_type": r["true_fault"],
                    "citation_validity": 0.0, "fault_specificity": 0.0,
                    "actionability": 0.0, "causal_soundness": 0.0,
                    "reference_valid": r["citation_correct"],
                }, ensure_ascii=False) + "\n")

    summary = summarise(records)
    with open(out / "track_c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for c, s in summary.items():
        logger.info("Condition %s: citation_correct=%d/%d | when classified correctly=%d/%d | "
                    "valid vs predicted row=%d/%d | missed=%d | templates=%d",
                    c[1], s["citation_correct"], s["n"], s["citation_correct_when_classified"],
                    s["correctly_classified"], s["reference_valid_vs_predicted"], s["explained"],
                    s["missed_by_detector"], s["template_fallbacks"])
    return summary


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
        help="directory for output files (explanations, scores template, summary; default: data/processed)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="path to config.yaml",
    )
    parser.add_argument(
        "--n-per-fault", type=int, default=cfg.get("evaluation", {}).get("samples_per_fault", 3),
        help="samples per true fault type (default: from config)",
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
