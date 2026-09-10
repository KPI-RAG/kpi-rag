"""
build_rca_evidence.py
---------------------
KPI-RAG RCA Layer — Evidence Construction Module

Reads Layer 2 predictions (JSONL) and the alignment table (JSON),
and produces a structured RCA evidence object for each window.

The RCA layer is an evidence construction layer, NOT an LLM layer.
It does NOT retrieve tickets and does NOT call the LLM.
It produces a clean RCA evidence JSON that the downstream RAG+LLM
module consumes alongside the retrieved tickets.

Input
-----
  layer2_predictions.jsonl   — one JSON object per line, each with:
      window_index
      predicted_fault_type
      confidence
      shap_top3               — list of top SHAP channel dicts
      signal_statistics       — dict of channel stat keys (mean/std/min/max)
      ground_truth_anomaly_type  (evaluation only — NOT used by RCA)

  alignment_table_v1.2.json  — the validated alignment table

Output
------
  rca_evidence.jsonl         — one RCA evidence object per window

RCA evidence object structure
------------------------------
{
  "window_index": int,
  "predicted_fault": str,
  "confidence": float,

  "rca_source": "alignment_table" | "jamming_fallback" | "unknown_fault",

  "kpi_evidence": [
    {
      "kpi": str,
      "expected": bool,
      "observed": bool,
      "shap_supported": bool,
      "shap_value": float | null,
      "shap_direction": str | null,
      "observed_mean": float | null,
      "evidence_status": "strong" | "supporting" | "missing" | "unexpected"
    },
    ...
  ],

  "coverage_summary": {
    "total_expected": int,
    "shap_covered": int,
    "observed": int,
    "coverage_ratio": float,
    "strong_count": int,
    "supporting_count": int,
    "missing_count": int,
    "unexpected_count": int
  },

  "root_cause": str,           # fault name as causal label
  "onset_pattern": str,        # from alignment table
  "causal_mechanism": str,     # from alignment table
  "causal_mechanism_sources": list[str],

  "standards_evidence": {
    "3gpp_reference": str,
    "3gpp_release": str,
    "3gpp_version": dict,
    "clause_text": str,
    "relation_type": str,
    "support_class": str,
    "evidence_strength": str,
    "kg_evidence_summary": str,
    "oran_component": str,
    "oran_component_rationale": str,
    "oran_spec_reference": str,
    "validation_status": str
  },

  "layer_a_observational": {   # raw signal stats from Layer 2
    "channel": float, ...
  },
  "layer_b_model_attribution": [  # SHAP attribution from Layer 2
    { "feature": str, "shap_value": float, "feature_vs_normal": str }, ...
  ],
  "layer_c_domain_standards": {   # alignment table grounding
    "causal_mechanism": str,
    "3gpp_reference": str,
    "oran_component": str
  },

  "_evaluation_note": str      # notes ground_truth is excluded from RCA
}

Usage
-----
  python build_rca_evidence.py \\
      --layer2  path/to/layer2_predictions.jsonl \\
      --table   path/to/alignment_table_v1.2.json \\
      --output  path/to/rca_evidence.jsonl \\
      [--log-level DEBUG]

  or import and call build_rca_for_window() directly.
"""

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
    assess_kpi_evidence,
    coverage_summary,
    extract_standards_evidence,
)

logger = logging.getLogger(__name__)


# ── Core RCA builder ──────────────────────────────────────────────────────────

def build_rca_for_window(
    window: dict[str, Any],
    table:  dict[str, Any],
) -> dict[str, Any]:
    """
    Build one RCA evidence object for a single Layer 2 prediction window.

    Parameters
    ----------
    window : dict
        A single Layer 2 prediction dict (one line of the JSONL).
    table : dict
        The alignment table index returned by load_alignment_table().

    Returns
    -------
    dict — the RCA evidence object (see module docstring for schema).
    """
    window_index   = window.get("window_index", -1)
    predicted      = window.get("predicted_fault_type", "")
    confidence     = window.get("confidence")
    shap_top3      = window.get("shap_top3", [])
    signal_stats   = window.get("signal_statistics", {})

    # ── Base structure ────────────────────────────────────────────────────────
    rca: dict[str, Any] = {
        "window_index":    window_index,
        "predicted_fault": predicted,
        "confidence":      confidence,
        # ground_truth deliberately excluded from RCA output.
        # It is available in the Layer 2 file for evaluation only.
        "_evaluation_note": (
            "ground_truth_anomaly_type is present in the Layer 2 file "
            "but is intentionally excluded from the RCA evidence object. "
            "RCA operates on predicted_fault_type only, as ground truth "
            "is unavailable in real deployment."
        ),
    }

    # ── Layer A: Observational evidence (raw signal statistics) ───────────────
    # Pass through the full signal_statistics block as-is.
    # The LLM and the evaluator can inspect any channel's mean/std/min/max.
    rca["layer_a_observational"] = signal_stats

    # ── Layer B: Model attribution (SHAP from Layer 2) ────────────────────────
    # Pass through SHAP exactly as received from Layer 2.
    # IMPORTANT: SHAP = model attribution, NOT physical causality.
    # Causal explanation comes from Layer C (alignment table).
    rca["layer_b_model_attribution"] = shap_top3

    # ── Jamming fallback ──────────────────────────────────────────────────────
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
        # KPI evidence still computed — SHAP and observed stats are meaningful
        # even for Jamming (they explain what the model saw)
        expected_kpis = []  # no expected KPIs from alignment table for Jamming
        ev = assess_kpi_evidence(expected_kpis, shap_top3, signal_stats)
        rca["kpi_evidence"]     = ev
        rca["coverage_summary"] = coverage_summary(ev)
        return rca

    # ── Look up alignment table ───────────────────────────────────────────────
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
        rca["kpi_evidence"]     = assess_kpi_evidence([], shap_top3, signal_stats)
        rca["coverage_summary"] = coverage_summary(rca["kpi_evidence"])
        return rca

    # ── Layer C: Domain and standards grounding (alignment table) ─────────────
    expected_kpis = row.get("expected_kpis", [])
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

    # ── Three-way KPI evidence assessment ─────────────────────────────────────
    # expected_kpis  = domain prior from alignment table
    # shap_top3      = model attribution from Layer 2
    # signal_stats   = actual observations from Layer 2
    ev = assess_kpi_evidence(expected_kpis, shap_top3, signal_stats)
    rca["kpi_evidence"]     = ev
    rca["coverage_summary"] = coverage_summary(ev)

    # ── Alignment table metadata (pass-through for provenance) ────────────────
    rca["alignment_table_row_id"]      = row.get("row_id")
    rca["alignment_table_domain"]      = row.get("domain")
    rca["alignment_table_validation"]  = row.get("validation_status")
    rca["alignment_table_notes"]       = row.get("notes")

    # ── SHAP caution note ─────────────────────────────────────────────────────
    # Remind downstream consumers that SHAP gives model attribution,
    # not physical causality. Physical causality comes from causal_mechanism.
    rca["shap_interpretation_note"] = (
        "SHAP values reflect model attribution (which features drove the "
        "prediction) not physical causality. Physical causal explanation "
        "is provided in causal_mechanism from the alignment table."
    )

    return rca


# ── Batch processor ───────────────────────────────────────────────────────────

def process_layer2_file(
    layer2_path:   Path,
    table:         dict[str, Any],
    output_path:   Path,
) -> dict[str, int]:

    counts = {
        "total":           0,
        "alignment_table": 0,
        "jamming_fallback": 0,
        "unknown_fault":   0,
        "errors":          0,
    }

    with layer2_path.open(encoding="utf-8") as fin:
        windows = json.load(fin)

    # Accept either a list of windows or a dict with a "windows" key
    if isinstance(windows, dict):
        windows = windows.get("windows", [])

    results = []
    for i, window in enumerate(windows):
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

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPI-RAG RCA Layer — build structured RCA evidence from Layer 2 predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer2",
        required=True,
        type=Path,
        help="Path to Layer 2 predictions JSONL file.",
    )
    parser.add_argument(
        "--table",
        required=True,
        type=Path,
        help="Path to Layer 2 predictions JSON file (list of window dicts).",
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
    logger.info("  Total windows processed : %d", counts["total"])
    logger.info("  alignment_table source  : %d", counts.get("alignment_table", 0))
    logger.info("  jamming_fallback source : %d", counts.get("jamming_fallback", 0))
    logger.info("  unknown_fault source    : %d", counts.get("unknown_fault", 0))
    logger.info("  errors                  : %d", counts["errors"])
    logger.info("  Output written to       : %s", args.output)

    if counts["errors"] > 0:
        logger.warning("%d windows failed — inspect logs above.", counts["errors"])
        sys.exit(1)


if __name__ == "__main__":
    main()
