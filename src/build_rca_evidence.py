from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rca_utils import (
    load_alignment_table,
    lookup_fault,
    is_jamming,
    extract_standards_evidence,
)

logger = logging.getLogger(__name__)

def build_rca_coverage_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize KPI evidence using model-derived affected_kpis as the profile."""
    total = len(evidence)
    observed = sum(1 for e in evidence if e.get("observed"))
    shap = sum(1 for e in evidence if e.get("shap_supported"))
    model = sum(1 for e in evidence if e.get("model_affected"))
    dataset = sum(1 for e in evidence if e.get("dataset_expected"))
    strong = sum(1 for e in evidence if e.get("evidence_status") == "strong")
    supporting = sum(1 for e in evidence if e.get("evidence_status") == "supporting")
    missing = sum(1 for e in evidence if e.get("evidence_status") in {"missing", "missing_observation"})
    unexpected = sum(
        1 for e in evidence
        if e.get("evidence_status") == "unexpected"
    )
    return {
        "total_kpis": total,
        "observed_kpis": observed,
        "shap_covered_kpis": shap,
        "model_affected_kpis": model,
        "dataset_expected_kpis": dataset,
        "coverage_ratio": model / total if total else 0.0,
        "shap_coverage_ratio": shap / observed if observed else 0.0,
        "strong_count": strong,
        "supporting_count": supporting,
        "missing_count": missing,
        "unexpected_count": unexpected,
    }




def _channel_from_key(key: str) -> str | None:
    for suffix in ("_mean", "_std", "_min", "_max"):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def _observed_channels(signal_stats: dict[str, Any]) -> set[str]:
    return {c for k in signal_stats for c in [_channel_from_key(k)] if c}


def _shap_by_channel(shap_top3: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for item in shap_top3:
        channel = item.get("channel") or _channel_from_key(item.get("feature", ""))
        if not channel:
            continue
        try:
            mag = abs(float(item.get("shap_value", 0)))
        except (TypeError, ValueError):
            mag = -1
        if channel not in out or mag > out[channel]["_abs_shap"]:
            out[channel] = {**item, "_abs_shap": mag}
    return out


def assess_kpi_evidence(
    signal_stats: dict[str, Any],
    shap_top3: list[dict[str, Any]],
    affected_kpis: list[str],
    expected_kpis: list[str],
) -> list[dict[str, Any]]:
    """Instance-first KPI evidence; expected_kpis is reference-only."""
    observed = _observed_channels(signal_stats)
    shap = _shap_by_channel(shap_top3)
    affected = set(affected_kpis or [])
    expected = set(expected_kpis or [])
    all_kpis = sorted(observed | set(shap) | affected | expected)
    result=[]
    for kpi in all_kpis:
        si=shap.get(kpi)
        obs= kpi in observed
        shap_present = si is not None
        shap_supports_prediction = (
            shap_present and si.get("shap_effect") == "toward_predicted_class"
        )
        model= kpi in affected
        ref= kpi in expected
        if obs and shap_supports_prediction and model: status="strong"
        elif obs and (shap_supports_prediction or model): status="supporting"
        elif obs and shap_present: status="contradictory"
        elif obs and ref: status="contextual"
        elif obs: status="observed_only"
        else: status="missing_observation"
        feature=si.get("feature") if si else None
        result.append({
            "kpi":kpi,
            "observed":obs,
            "observed_mean":signal_stats.get(f"{kpi}_mean"),
            "observed_feature_value":signal_stats.get(feature) if feature else None,
            "shap_present":shap_present,
            "shap_supported":shap_supports_prediction,
            "shap_value":si.get("shap_value") if si else None,
            "shap_direction":si.get("feature_vs_normal") if si else None,
            "shap_effect":si.get("shap_effect") if si else None,
            "shap_consistency": (
                "supports_prediction" if shap_supports_prediction else
                "contradicts_prediction" if shap_present else
                "not_available"
            ),
            "shap_feature":feature,
            "model_affected":model,
            "dataset_expected":ref,
            "evidence_status":status,
        })
    return result


def evidence_summary(ev):
    observed=sum(e["observed"] for e in ev)
    shap=sum(e["shap_supported"] for e in ev)
    affected=sum(e["model_affected"] for e in ev)
    return {
        "total_kpis":len(ev), "observed_kpis":observed,
        "shap_covered_kpis":shap, "model_affected_kpis":affected,
        "dataset_expected_kpis":sum(e["dataset_expected"] for e in ev),
        "strong_count":sum(e["evidence_status"]=="strong" for e in ev),
        "supporting_count":sum(e["evidence_status"]=="supporting" for e in ev),
        "contextual_count":sum(e["evidence_status"]=="contextual" for e in ev),
        "observed_only_count":sum(e["evidence_status"]=="observed_only" for e in ev),
        "contradictory_count":sum(e["evidence_status"]=="contradictory" for e in ev),
        "missing_observation_count":sum(e["evidence_status"]=="missing_observation" for e in ev),
    }

# Core RCA builder 
def build_rca_for_window(
    window: dict[str, Any],
    table:  dict[str, Any],
) -> dict[str, Any]:
    """
    Build one RCA evidence object for a single Layer 2 prediction window.

    Parameters
    ----------
    window : dict
        A single Layer 2 prediction dictionary from the JSON array.
    table : dict
        The alignment table index returned by load_alignment_table().

    Returns
    -------
    dict — the RCA evidence object (see module docstring for schema).
    """
    window_index   = window.get("window_index", -1)
    predicted      = window.get("anomaly_type", "")
    confidence     = window.get("confidence")
    shap_top3      = window.get("shap_top3", [])
    signal_stats   = window.get("signal_statistics", {})

    # Base structure 
    rca: dict[str, Any] = {
        "window_index":    window_index,
        "predicted_fault": predicted,
        "confidence":      confidence,
        # ground_truth deliberately excluded from RCA output.
        # It is available in the Layer 2 file for evaluation only.
        "_evaluation_note": (
            "ground-truth labels may be present in the Layer 2 file "
            "but is intentionally excluded from the RCA evidence object. "
            "RCA operates on anomaly_type only, as ground truth "
            "is unavailable in real deployment."
        ),
    }

    # Layer A: Observational evidence (raw signal statistics) 
    rca["layer_a_observational"] = signal_stats

    # Layer B: Model attribution (SHAP from Layer 2)
    rca["layer_b_model_attribution"] = shap_top3

    # Jamming fallback 
    if is_jamming(predicted):
        jamming_note = table.get("_jamming_note", {})
        rca["rca_source"]       = "jamming_fallback"
        rca["root_cause"]       = predicted
        rca["onset_pattern"]    = (
            "Instantaneous — jamming is an adversarial RF attack "
            "with no gradual build-up."
        )
        rca["causal_mechanism"] = (
            "Jamming is a physical RF attack on the radio spectrum. "
            "No corresponding 3GPP Technical Specification clause exists. "
            "3GPP standards govern legitimate network behavior only."
        )
        rca["causal_mechanism_sources"] = []
        rca["standards_evidence"]       = None
        rca["layer_c_domain_standards"] = {
            "causal_mechanism": rca["causal_mechanism"],
            "3gpp_reference":   "None — physical RF attack",
            "oran_component":   None,
        }
        rca["pipeline_fallback"] = jamming_note.get(
            "pipeline_fallback",
            "Ticket-retrieval evidence only. Alignment table not applicable.",
        )
        affected_kpis = []
        expected_kpis = []
        ev = assess_kpi_evidence(signal_stats, shap_top3, [], [])
        rca["kpi_evidence"] = ev
        rca["coverage_summary"] = build_rca_coverage_summary(ev)
        rca["alignment_table_expected_kpis"] = []
        rca["alignment_table_affected_kpis"] = []
        return rca

    # Look up alignment table 
    row = lookup_fault(table, predicted)

    if row is None:
        # Fault not in alignment table and not Jamming
        logger.warning(
            "Window %d: predicted fault '%s' not found in alignment table.",
            window_index,
            predicted,
        )
        rca["rca_source"]       = "unknown_fault"
        rca["root_cause"]       = predicted
        rca["onset_pattern"]    = None
        rca["causal_mechanism"] = None
        rca["causal_mechanism_sources"] = []
        rca["standards_evidence"]       = None
        rca["layer_c_domain_standards"] = None
        ev = assess_kpi_evidence(signal_stats, shap_top3, [], [])
        rca["kpi_evidence"] = ev
        rca["coverage_summary"] = build_rca_coverage_summary(ev)
        return rca

    # Layer C: Domain and standards grounding (alignment table) 
    expected_kpis = row.get("expected_kpis", [])
    affected_kpis = row.get("affected_kpis", [])
    standards     = extract_standards_evidence(row)

    rca["rca_source"]       = "alignment_table"
    rca["root_cause"]       = row["telecomts_fault"]
    rca["onset_pattern"]    = row.get("onset_pattern")
    rca["causal_mechanism"] = row.get("causal_mechanism")
    rca["causal_mechanism_sources"] = row.get("causal_mechanism_sources", [])
    rca["standards_evidence"]       = standards
    rca["layer_c_domain_standards"] = {
        "causal_mechanism": row.get("causal_mechanism"),
        "3gpp_reference":   row.get("3gpp_reference"),
        "oran_component":   row.get("oran_component"),
    }

    ev = assess_kpi_evidence(signal_stats, shap_top3, affected_kpis, expected_kpis)

    rca["kpi_evidence"]     = ev
    rca["coverage_summary"] = build_rca_coverage_summary(ev)
    rca["alignment_table_expected_kpis"] = expected_kpis
    rca["alignment_table_affected_kpis"] = affected_kpis
    rca["kpi_evidence_interpretation"] = (
        "signal_statistics are the primary instance-level observations; "
        "shap_top3 provides window-level model attribution; affected_kpis "
        "is the model-derived class-level KPI profile; expected_kpis is "
        "retained only as a TelecomTS reference profile and does not drive "
        "the RCA evidence assessment."
    )

    # Alignment table metadata (pass-through for provenance) 
    rca["alignment_table_row_id"]      = row.get("row_id")
    rca["alignment_table_domain"]      = row.get("domain")
    rca["alignment_table_validation"]  = row.get("validation_status")
    rca["alignment_table_notes"]       = row.get("notes")

    # SHAP caution note 
    rca["shap_interpretation_note"] = (
        "SHAP values reflect model attribution (which features drove the "
        "prediction) not physical causality. Physical causal explanation "
        "is provided in causal_mechanism from the alignment table."
    )

    return rca


# Batch processor 
def process_layer2_file(
    layer2_path:   Path,
    table:         dict[str, Any],
    output_path:   Path,
) -> dict[str, int]:

    counts = {
    "input_windows": 0,
    "normal_skipped": 0,
    "jamming_windows": 0,
    "total": 0,
    "alignment_table": 0,
    "jamming_fallback": 0,
    "unknown_fault": 0,
    "errors": 0,
}
    
    with layer2_path.open(encoding="utf-8") as fin:
        windows = json.load(fin)

    counts["input_windows"] = len(windows) if isinstance(windows, list) else 0

    # Accept either a list of windows or a dict with a "windows" key
    if isinstance(windows, dict):
        windows = windows.get("windows", [])

    results = []
    counts["input_windows"] = len(windows)
    for i, window in enumerate(windows):
        predicted = str(window.get("anomaly_type", "")).strip()
    
        if predicted.lower() == "normal":
            counts["normal_skipped"] += 1
            continue
    
        # Exclude Jamming from RCA evidence, but count every skipped window.
        if is_jamming(predicted):
            counts["jamming_windows"] += 1
            logger.info(
                "Skipping Jamming window %s",
                window.get("window_index", i),
            )
            continue

        try:
            rca = build_rca_for_window(window, table)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Window index %s: RCA build error — %s",
                window.get("window_index", i),
                exc,
                exc_info=True,
            )
            counts["errors"] += 1
            continue

        results.append(rca)
        counts["total"] += 1
        source = rca.get("rca_source", "unknown_fault")
        counts[source] = counts.get(source, 0) + 1

        if counts["total"] % 100 == 0:
            logger.info("Processed %d windows ...", counts["total"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    return counts

# CLI 
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPI-RAG RCA Layer — build structured RCA evidence from Layer 2 predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer2",
        required=True,
        type=Path,
        help="Path to Layer 2 predictions JSON file.",
    )
    parser.add_argument(
        "--table",
        required=True,
        type=Path,
        help="Path to alignment table JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write RCA evidence JSON output (list of RCA dicts).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading alignment table: %s", args.table)
    table = load_alignment_table(args.table)

    logger.info("Processing Layer 2 predictions: %s", args.layer2)

    counts = process_layer2_file(args.layer2, table, args.output)

  
    logger.info("Done.")
    logger.info("  Input windows             : %d", counts.get("input_windows", 0))
    logger.info("  Normal windows skipped    : %d", counts.get("normal_skipped", 0))
    logger.info("  Jamming windows skipped   : %d", counts.get("jamming_windows", 0))
    logger.info("  Anomalous RCA records     : %d", counts.get("total", 0))
    logger.info("  Alignment table source    : %d", counts.get("alignment_table", 0))
    logger.info("  Jamming fallback source   : %d", counts.get("jamming_fallback", 0))
    logger.info("  Unknown fault source      : %d", counts.get("unknown_fault", 0))
    logger.info("  Errors                    : %d", counts.get("errors", 0))
    logger.info("  Output written to         : %s", args.output)

    if counts["errors"] > 0:
        logger.warning(
            "%d windows failed — inspect logs above.",
            counts["errors"],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
