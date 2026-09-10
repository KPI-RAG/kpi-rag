"""
rca_utils.py
------------
Utility functions for the KPI-RAG RCA layer.

Responsibilities:
  - Load and index the alignment table by fault name
  - Assess KPI evidence (expected vs observed vs SHAP)
  - Derive KPI direction labels from signal_statistics
  - Handle the Jamming fallback

Does NOT:
  - Retrieve tickets
  - Call an LLM
  - Perform RAG
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Evidence status labels ────────────────────────────────────────────────────
STRONG       = "strong"        # expected + observed + top SHAP
SUPPORTING   = "supporting"    # expected + observed, not in top SHAP
MISSING      = "missing"       # expected but not in signal_statistics
UNEXPECTED   = "unexpected"    # in top SHAP but not expected for this fault


# ── Alignment table loading ───────────────────────────────────────────────────

def load_alignment_table(path: str | Path) -> dict[str, Any]:
    """
    Load alignment_table_v1.2.json and return a dict keyed by
    telecomts_fault (lower-stripped) for fast lookup.

    Returns
    -------
    {
        "co-channel interference (severe)": { ...row dict... },
        ...
        "_jamming_note": { ...jamming fallback dict... },
        "_meta": { version, status, relation_type_definitions, ... }
    }
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    index: dict[str, Any] = {}

    for row in raw.get("rows", []):
        key = row["telecomts_fault"].strip().lower()
        index[key] = row

    # Store jamming fallback and top-level metadata separately
    index["_jamming_note"] = raw.get("jamming_note", {})
    index["_meta"] = {
        "version":                  raw.get("version"),
        "status":                   raw.get("status"),
        "relation_type_definitions": raw.get("relation_type_definitions", {}),
        "classification_scheme":    raw.get("classification_scheme", {}),
        "shap_update_note":         raw.get("shap_update_note"),
    }

    logger.info(
        "Alignment table loaded: %d fault rows (version=%s, status=%s)",
        len(raw.get("rows", [])),
        raw.get("version"),
        raw.get("status"),
    )
    return index


def lookup_fault(table: dict[str, Any], predicted_fault: str) -> dict[str, Any] | None:
    """
    Look up a fault row by predicted_fault string (case-insensitive).
    Returns None if not found (caller must handle Jamming fallback separately).
    """
    key = predicted_fault.strip().lower()
    row = table.get(key)
    if row is None:
        logger.warning("Fault '%s' not found in alignment table.", predicted_fault)
    return row


def is_jamming(predicted_fault: str) -> bool:
    return "jamming" in predicted_fault.strip().lower()


# ── KPI evidence assessment ───────────────────────────────────────────────────

def assess_kpi_evidence(
    expected_kpis: list[str],
    shap_top3: list[dict[str, Any]],
    signal_statistics: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a three-way KPI evidence assessment:
        expected  = from alignment table (domain prior)
        observed  = present in signal_statistics from Layer 2
        shap      = appears in shap_top3 from Layer 2

    Returns a list of evidence dicts, one per unique KPI across all
    three sources, sorted by evidence strength (strong first).

    Parameters
    ----------
    expected_kpis : list[str]
        From alignment_table row["expected_kpis"].
    shap_top3 : list[dict]
        Each dict has at minimum: "feature" (channel name), "shap_value",
        optionally "feature_vs_normal".
    signal_statistics : dict
        The signal_statistics block from the Layer 2 JSON, e.g.
        {"RSRP_mean": ..., "UL_SNR_mean": ..., ...}

    Returns
    -------
    [
        {
            "kpi": "UL_SNR",
            "expected": True,
            "observed": True,
            "shap_supported": True,
            "shap_value": -0.84,
            "shap_direction": "above_normal_mean",   # from shap_top3 if present
            "observed_mean": 20.72,                  # from signal_statistics if present
            "evidence_status": "strong"
        },
        ...
    ]
    """
    # Index SHAP entries by channel name (strip suffixes like _mean, _max etc.)
    shap_index: dict[str, dict] = {}
    for entry in shap_top3:
        # Use "channel" (base KPI name e.g. "RSRP") when present,
        # falling back to "feature" (stat variant e.g. "RSRP_max").
        # This ensures SHAP entries match expected_kpis which use base names.
        channel = entry.get("channel") or entry.get("feature", "")
        # Store under base channel name; keep full entry for detail fields
        shap_index[channel] = entry

    # Collect all unique KPI base names from all three sources
    def _base(name: str) -> str:
        """Strip stat suffixes so RSRP_mean → RSRP."""
        for suffix in ("_mean", "_std", "_min", "_max"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    stat_bases = {_base(k) for k in signal_statistics}

    all_kpis: set[str] = set(expected_kpis) | set(shap_index.keys()) | stat_bases

    evidence: list[dict[str, Any]] = []

    for kpi in sorted(all_kpis):
        exp    = kpi in expected_kpis
        shap_e = kpi in shap_index
        obs    = kpi in stat_bases

        # Determine evidence_status
        if exp and obs and shap_e:
            status = STRONG
        elif exp and obs and not shap_e:
            status = SUPPORTING
        elif exp and not obs:
            status = MISSING
        elif shap_e and not exp:
            status = UNEXPECTED
        else:
            status = SUPPORTING  # observed but neither expected nor SHAP

        entry: dict[str, Any] = {
            "kpi":            kpi,
            "expected":       exp,
            "observed":       obs,
            "shap_supported": shap_e,
            "evidence_status": status,
        }

        # Attach SHAP details if available
        if shap_e:
            shap_entry = shap_index[kpi]
            entry["shap_value"]     = shap_entry.get("shap_value")
            entry["shap_direction"] = shap_entry.get("feature_vs_normal")
            entry["shap_effect"]    = shap_entry.get("shap_effect")
            entry["shap_feature"]   = shap_entry.get("feature")  # e.g. "RSRP_max"

        # Attach observed mean if available
        mean_key = f"{kpi}_mean"
        if mean_key in signal_statistics:
            entry["observed_mean"] = signal_statistics[mean_key]

        evidence.append(entry)

    # Sort: strong → supporting → missing → unexpected
    _order = {STRONG: 0, SUPPORTING: 1, MISSING: 2, UNEXPECTED: 3}
    evidence.sort(key=lambda x: _order.get(x["evidence_status"], 9))

    return evidence


def coverage_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics over the KPI evidence list.

    Returns
    -------
    {
        "total_expected": 5,
        "shap_covered":   3,
        "observed":       5,
        "coverage_ratio": 0.6,   # shap_covered / total_expected
        "strong_count":   3,
        "supporting_count": 2,
        "missing_count":  0,
        "unexpected_count": 1
    }
    """
    expected  = [e for e in evidence if e["expected"]]
    shap_cov  = [e for e in expected if e["shap_supported"]]

    total_exp = len(expected)
    ratio     = len(shap_cov) / total_exp if total_exp > 0 else 0.0

    counts = {s: 0 for s in (STRONG, SUPPORTING, MISSING, UNEXPECTED)}
    for e in evidence:
        counts[e["evidence_status"]] = counts.get(e["evidence_status"], 0) + 1

    return {
        "total_expected":   total_exp,
        "shap_covered":     len(shap_cov),
        "observed":         sum(1 for e in evidence if e["observed"]),
        "coverage_ratio":   round(ratio, 3),
        "strong_count":     counts[STRONG],
        "supporting_count": counts[SUPPORTING],
        "missing_count":    counts[MISSING],
        "unexpected_count": counts[UNEXPECTED],
    }


# ── Standards evidence extraction ─────────────────────────────────────────────

def extract_standards_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the standards-grounding fields from an alignment table row
    into a clean sub-object for the RCA output.
    """
    return {
        "3gpp_reference":   row.get("3gpp_reference"),
        "3gpp_release":     row.get("3gpp_release"),
        "3gpp_version":     row.get("3gpp_version"),
        "clause_text":      row.get("clause_text"),
        "relation_type":    row.get("relation_type"),
        "support_class":    row.get("support_class"),
        "evidence_strength": row.get("evidence_strength"),
        "kg_evidence_summary": row.get("kg_evidence_summary"),
        "oran_component":       row.get("oran_component"),
        "oran_component_rationale": row.get("oran_component_rationale"),
        "oran_spec_reference":  row.get("oran_spec_reference"),
        "validation_status":    row.get("validation_status"),
    }
