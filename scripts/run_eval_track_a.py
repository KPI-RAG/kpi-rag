"""Track A: detection / classification metrics recomputed from the Phase 2 predictions,
plus retrieval precision for the flagged windows. Writes one JSON file.

Layer 2 itself runs offline (see src/ml_model.py), so this re-derives the metrics
from layer2_output_sessionsplit_phase2.json + ground truth instead of re-running models.
"""
import argparse
import json
import logging
import os
import statistics

from src.config_loader import load_config
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Same grouping as Farida's validation.json (class_grouping_reporting_complement)
CONGESTION_FAMILY = {
    "Buffer Overflow (Gradual Buildup)",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
}


def _family(label: str) -> str:
    return "congestion_family" if label in CONGESTION_FAMILY else label


def compute_cascade_metrics(preds: dict[int, str], truth: dict[int, str]) -> dict:
    """preds: window -> predicted label ("Normal" or a fault); truth: window -> true fault (faults only)."""
    faults = [w for w in preds if w in truth]
    flagged = [w for w in preds if preds[w] != "Normal"]
    tp = [w for w in flagged if w in truth]
    exact = sum(preds[w] == truth[w] for w in faults)
    family = sum(_family(preds[w]) == _family(truth[w]) for w in faults)

    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    return {
        "test_windows": len(preds),
        "true_faults": len(faults),
        "flagged_as_fault": len(flagged),
        "true_faults_detected": len(tp),
        "false_alarms": len(flagged) - len(tp),
        "binary_recall": ratio(len(tp), len(faults)),
        "binary_precision": ratio(len(tp), len(flagged)),
        "exact_fault_type_correct": exact,
        "exact_accuracy_on_all_true_faults": ratio(exact, len(faults)),
        "exact_accuracy_on_detected_faults": ratio(exact, len(tp)),
        "family_accuracy_on_all_true_faults": ratio(family, len(faults)),
        # Jamming (one real OTA anomaly) is 94 of the 300 test faults; report the rest separately
        "exact_accuracy_excluding_jamming": ratio(
            sum(preds[w] == truth[w] for w in faults if truth[w] != "Jamming"),
            sum(truth[w] != "Jamming" for w in faults)),
        "binary_recall_excluding_jamming": ratio(
            sum(preds[w] != "Normal" for w in faults if truth[w] != "Jamming"),
            sum(truth[w] != "Jamming" for w in faults)),
    }


def compute_retrieval_metrics(records: list[dict], truth: dict[int, str], cfg: dict) -> dict:
    """System retrieval (filtered to the predicted class) plus a label-free diagnostic.

    The diagnostic drops the class filter: P@k of the retrieved incidents' fault type
    vs the TRUE label measures how much the KPI evidence alone identifies the fault.
    """
    from src.kg_indexer import get_collection
    from src.rag_query import build_query, query_from_classifier_output, retrieve
    from src.schema import ClassifierOutput

    collection = get_collection(cfg)
    rag = cfg["rag"]
    system, unfiltered = [], []
    for r in records:
        payload = ClassifierOutput(**r)
        tickets, low_conf = query_from_classifier_output(payload, collection, cfg)
        system.append((tickets[0].similarity_score if tickets else 0.0, low_conf))
        free, _ = retrieve(build_query(payload), collection, rag["embedding_model"], k=rag["top_k"],
                           threshold=rag["cosine_threshold"])
        if r["window_index"] in truth:
            unfiltered.append((truth[r["window_index"]], [t.anomaly_type for t in free]))

    def precision(pairs, k):
        return statistics.mean(sum(t == label for t in types[:k]) / k for label, types in pairs) if pairs else None

    true_labels = [label for label, _ in unfiltered]
    top1 = [s for s, _ in system]
    return {
        "queries": len(system),
        "system_top1_similarity_min": min(top1),
        "system_top1_similarity_median": statistics.median(top1),
        "system_top1_similarity_max": max(top1),
        "system_low_confidence_flags": sum(lc for _, lc in system),
        "diagnostic_queries_on_true_faults": len(unfiltered),
        "diagnostic_label_free_p_at_1_vs_true": precision(unfiltered, 1),
        "diagnostic_label_free_p_at_5_vs_true": precision(unfiltered, 5),
        "diagnostic_majority_class_share": (max(true_labels.count(c) for c in set(true_labels)) / len(true_labels)
                                            if true_labels else None),
    }


def run_track_a(cfg: dict, output_path: str, with_retrieval: bool = True) -> dict:
    data_cfg = cfg["data"]
    processed = data_cfg["processed_path"]
    with open(os.path.join(processed, data_cfg["layer2_output_file"]), encoding="utf-8") as f:
        records = json.load(f)
    with open(os.path.join(processed, "layer2_rag_handoff_sessionsplit.json"), encoding="utf-8") as f:
        handoff = json.load(f)

    preds = {r["window_index"]: r["anomaly_type"] for r in records}
    # the handoff file holds every anomalous window; keep the ones in this (test) prediction set
    truth = {r["window_index"]: r["ground_truth_anomaly_type"] for r in handoff if r["window_index"] in preds}

    results = {"cascade": compute_cascade_metrics(preds, truth)}
    if with_retrieval:
        flagged = [r for r in records if r["anomaly_type"] != "Normal"]
        results["retrieval"] = compute_retrieval_metrics(flagged, truth, cfg)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for section, values in results.items():
        for k, v in values.items():
            logger.info("%s.%s = %s", section, k, f"{v:.4f}" if isinstance(v, float) else v)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute Track A + retrieval metrics")
    parser.add_argument("--output", default="data/processed/track_a_metrics.json")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-retrieval", action="store_true", help="skip the ChromaDB queries")
    args = parser.parse_args()
    setup_logging(__name__)
    run_track_a(load_config(args.config), args.output, with_retrieval=not args.no_retrieval)


if __name__ == "__main__":
    main()
