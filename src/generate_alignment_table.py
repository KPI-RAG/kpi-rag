from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



# Constants
# All 10 TelecomTS fault types (Jamming is excluded — no 3GPP clause applies)
FAULT_TYPES = [
    "Antenna Failure",
    "Buffer Overflow (Gradual Buildup)",
    "Co-Channel Interference (Mild)",
    "Co-Channel Interference (Severe)",
    "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)",
    "Faulty RF Filters (Temporal)",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
    "Resource Allocation Bugs",
]

# Stage 1: keyword sets per fault type used to score KG candidates
# These replicate the search vocabulary used by the src/rag/ scripts
FAULT_KG_KEYWORDS: dict[str, list[str]] = {
    "Antenna Failure": [
        "antenna", "antenna connector", "AAS BS", "AAS receiver",
        "receiver RF", "transmitter", "reference sensitivity", "RSRP",
        "signal degradation", "RF chain",
    ],
    "Buffer Overflow (Gradual Buildup)": [
        "buffer status report", "BSR", "MAC CE", "uplink buffer",
        "PRB allocation", "buffer saturation", "scheduler",
        "periodicBSR", "retxBSR", "MAC-CellGroupConfig",
    ],
    "Co-Channel Interference (Mild)": [
        "co-channel interference", "in-channel selectivity", "SINR",
        "interference", "SNR", "coexistence", "guardband",
        "zero forcing", "UL SNR",
    ],
    "Co-Channel Interference (Severe)": [
        "co-channel interference", "in-channel selectivity", "OTA co-location",
        "blocking", "mutual interference", "SINR collapse", "SNR",
        "interference", "spurious emissions",
    ],
    "Doppler Shift (Severe)": [
        "Doppler", "channel aging", "time-varying channel", "UE velocity",
        "mobility", "channel model", "frequency shift", "dual mobility",
    ],
    "Faulty Handover Algorithm (Too Frequent)": [
        "handover", "HANDOVER REQUIRED", "ping-pong", "A3 event",
        "hysteresis", "conditional handover", "handover failure",
        "HANDOVER REQUEST", "cell reselection", "RRC",
    ],
    "Faulty RF Filters (Temporal)": [
        "RF filter", "out-of-band blocking", "spurious emission",
        "receiver blocking", "transmitter spurious", "stopband",
        "adjacent channel", "in-channel selectivity", "unwanted emission",
    ],
    "High Network Congestion (Gradual Buildup)": [
        "PRB usage", "PRB utilization", "resource block", "congestion",
        "throughput", "scheduler", "layer 2 measurement",
        "DL PRB", "UL PRB", "traffic load",
    ],
    "High Network Congestion (Sudden Spike)": [
        "PRB usage", "PRB utilization", "resource block", "traffic burst",
        "layer 2 measurement", "DL PRB", "UL PRB", "congestion",
    ],
    "Resource Allocation Bugs": [
        "resource allocation", "MAC scheduler", "scheduling", "PRB",
        "RRM", "radio resource management", "HARQ", "uplink grant",
        "downlink assignment", "BSR", "scheduler bug",
    ],
}

# Stage 2: O-RAN component keyword map — mirrors build_oran_alignment.py
# Stage 1: normalization map — KG CSVs use shorter names without the
# parenthetical qualifier.  Both the full TelecomTS name and the short
# CSV name are mapped to the same canonical key so load_kg_csv() hits.
_CSV_FAULT_ALIASES: dict[str, str] = {
    # CSV short name  →  canonical TelecomTS fault (used as table row key)
    "Antenna Failure":              "Antenna Failure",
    "Buffer Overflow":              "Buffer Overflow (Gradual Buildup)",
    "Co-Channel Interference":      "Co-Channel Interference (Mild)",   # also maps Severe
    "Doppler Shift":                "Doppler Shift (Severe)",
    "Faulty Handover Algorithm":    "Faulty Handover Algorithm (Too Frequent)",
    "Faulty RF Filters":            "Faulty RF Filters (Temporal)",
    "High Network Congestion":      "High Network Congestion (Gradual Buildup)",  # also Sudden Spike
    "Resource Allocation Bugs":     "Resource Allocation Bugs",
}

# Faults that share a CSV short name but map to TWO canonical rows
_CSV_FAULT_MULTI: dict[str, list[str]] = {
    "Co-Channel Interference": [
        "Co-Channel Interference (Mild)",
        "Co-Channel Interference (Severe)",
    ],
    "High Network Congestion": [
        "High Network Congestion (Gradual Buildup)",
        "High Network Congestion (Sudden Spike)",
    ],
}

ORAN_FAULT_KEYWORDS: dict[str, list[str]] = {
    "Antenna Failure": ["antenna", "rf", "receiver", "transmitter", "o-ru"],
    "Buffer Overflow (Gradual Buildup)": ["buffer", "bsr", "mac", "scheduler", "o-du"],
    "Co-Channel Interference (Mild)": ["interference", "sinr", "coexistence", "o-ru", "rf"],
    "Co-Channel Interference (Severe)": ["interference", "sinr", "blocking", "o-ru", "ric"],
    "Doppler Shift (Severe)": ["doppler", "mobility", "channel", "velocity", "o-ru"],
    "Faulty Handover Algorithm (Too Frequent)": ["handover", "mobility", "ric", "e2", "near-rt"],
    "Faulty RF Filters (Temporal)": ["filter", "rf", "spurious", "blocking", "o-ru"],
    "High Network Congestion (Gradual Buildup)": ["prb", "congestion", "scheduler", "load", "o-du"],
    "High Network Congestion (Sudden Spike)": ["prb", "burst", "scheduler", "congestion", "o-du"],
    "Resource Allocation Bugs": ["resource", "scheduler", "mac", "prb", "o-du", "ric"],
}



# Stage 1 — 3GPP KG evidence extraction
def load_kg_csv(kg_csv_dir: Path) -> dict[str, list[dict]]:
    """
    Read all relevant KG investigation CSVs from data/3GPP_KG/.

    Priority order (higher index = higher confidence):
      1. fault_coverage_candidates.csv  — broad search output
      2. targeted_fault_search.csv      — targeted keyword search
      3. fault_top_candidates.csv       — scored and ranked
      4. fault_evidence_review.csv      — manually reviewed evidence

    Returns a dict keyed by telecomts_fault → list of candidate rows.
    """
    candidates: dict[str, list[dict]] = defaultdict(list)

    csv_files = [
        ("fault_coverage_candidates.csv", 1),
        ("targeted_fault_search.csv", 2),
        ("fault_top_candidates.csv", 3),
        ("doppler_broad_search.csv", 3),   # Doppler-specific search output
        ("fault_evidence_review.csv", 4),
    ]

    for fname, weight in csv_files:
        fpath = kg_csv_dir / fname
        if not fpath.exists():
            logger.debug("KG CSV not found, skipping: %s", fpath)
            continue
        logger.info("Loading KG CSV (weight=%d): %s", weight, fpath)
        try:
            with fpath.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fault_raw = row.get("telecomts_fault", "").strip()
                    if not fault_raw:
                        continue
                    row["_source_file"] = fname
                    row["_source_weight"] = weight
                    # Resolve canonical fault names: a CSV short name may map
                    # to one or two canonical TelecomTS fault rows.
                    targets: list[str]
                    if fault_raw in _CSV_FAULT_MULTI:
                        targets = _CSV_FAULT_MULTI[fault_raw]
                    elif fault_raw in _CSV_FAULT_ALIASES:
                        targets = [_CSV_FAULT_ALIASES[fault_raw]]
                    else:
                        targets = [fault_raw]  # already canonical or unknown
                    for t in targets:
                        candidates[t].append(row)
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)

    logger.info(
        "Stage 1: loaded KG evidence for %d fault types across %d CSVs",
        len(candidates),
        sum(1 for _, w in csv_files if (kg_csv_dir / _).exists()),
    )
    return dict(candidates)


def extract_kg_summary(fault: str, rows: list[dict], top_n: int = 8) -> tuple[str, list[dict]]:
    """
    Score and summarise the KG candidate rows for one fault type.

    Returns:
        summary_str  — human-readable KG evidence summary
        top_rows     — top-N scored candidate dicts for the audit trail
    """
    keywords = [kw.lower() for kw in FAULT_KG_KEYWORDS.get(fault, [])]

    scored: list[tuple[float, dict]] = []
    for row in rows:
        text = " ".join(str(v) for v in row.values()).lower()
        base_score = float(row.get("candidate_score") or 0)
        weight = float(row.get("_source_weight", 1))
        keyword_hits = sum(1 for kw in keywords if kw in text)
        score = base_score + keyword_hits * 0.5 + weight * 0.3
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in scored[:top_n]]

    entity_names = []
    source_files: set[str] = set()
    for r in top:
        name = r.get("entity_id") or r.get("node_id") or ""
        src = r.get("source_file") or r.get("source_id") or ""
        if name and name not in entity_names:
            entity_names.append(name)
        if src:
            source_files.add(src.split("/")[-1].split(".")[0][:20])

    summary = (
        f"KG Stage-1 evidence ({len(rows)} candidate rows across "
        f"{len(set(r.get('_source_file','') for r in rows))} CSV files). "
        f"Top entities: {', '.join(entity_names[:6]) if entity_names else 'none'}. "
        f"Source specs: {', '.join(sorted(source_files)[:4]) if source_files else 'unknown'}."
    )

    audit_rows = [
        {
            "entity_id": r.get("entity_id") or r.get("node_id"),
            "entity_type": r.get("entity_type"),
            "description_snippet": (r.get("description") or "")[:120],
            "source_file": r.get("source_file") or r.get("source_id"),
            "matched_terms": r.get("matched_terms"),
            "candidate_score": r.get("candidate_score"),
            "_source_csv": r.get("_source_file"),
        }
        for r in top
    ]
    return summary, audit_rows



# Stage 2 — O-RAN KG evidence extraction
def load_oran_nodes(oran_kg_path: Path) -> list[dict]:
    """Load oran_relevant_nodes.jsonl into memory."""
    if not oran_kg_path.exists():
        logger.warning("O-RAN KG file not found: %s", oran_kg_path)
        return []
    nodes = []
    with oran_kg_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info("Stage 2: loaded %d O-RAN KG nodes from %s", len(nodes), oran_kg_path)
    return nodes


def score_oran_nodes(
    fault: str,
    nodes: list[dict],
    top_n: int = 5,
) -> tuple[str, list[dict]]:
    """
    Score O-RAN nodes against a fault type using keyword matching.
    Returns (summary_str, top_candidate_dicts).
    """
    keywords = [kw.lower() for kw in ORAN_FAULT_KEYWORDS.get(fault, [])]
    if not keywords:
        return "No O-RAN keywords defined for this fault type.", []

    scored: list[tuple[float, dict]] = []
    for node in nodes:
        text = " ".join(
            str(v) for v in [
                node.get("name", ""),
                node.get("description", ""),
                node.get("category", ""),
                node.get("spec_reference", ""),
                " ".join(node.get("labels", [])),
                " ".join(node.get("tags", [])),
            ]
        ).lower()
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scored.append((hits, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [n for _, n in scored[:top_n]]

    if not top:
        return f"No O-RAN KG nodes matched keywords for '{fault}'.", []

    names = [n.get("name") or n.get("id", "?") for n in top]
    specs = list({n.get("spec_reference", "") for n in top if n.get("spec_reference")})

    summary = (
        f"O-RAN KG Stage-2 evidence: {len(scored)} matching nodes. "
        f"Top components: {', '.join(names[:4])}. "
        f"Spec references: {', '.join(specs[:3]) if specs else 'none'}."
    )
    audit = [
        {
            "name": n.get("name"),
            "category": n.get("category"),
            "spec_reference": n.get("spec_reference"),
            "description_snippet": (n.get("description") or "")[:120],
        }
        for n in top
    ]
    return summary, audit



# Stage 3 — Row assembly
def load_base_table(base_path: Path) -> dict[str, Any]:
    """Load alignment_table_v1.0.json."""
    with base_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_shap_values(shap_path: Path | None) -> dict[str, list[str]]:
    """
    Load SHAP-ranked features if a shap file is provided.

    Expected format (one of):
      { "Antenna Failure": ["RSRP", "UL_SNR", ...], ... }
      or a list of { "fault_type": str, "ranked_features": [...] }

    Returns dict: fault_type → [ranked channel names]
    """
    if shap_path is None or not shap_path.exists():
        return {}
    with shap_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if isinstance(v, list)}
    if isinstance(raw, list):
        result = {}
        for item in raw:
            fault = item.get("fault_type") or item.get("telecomts_fault")
            features = item.get("ranked_features") or item.get("shap_top_features")
            if fault and features:
                result[fault] = features
        return result
    return {}


def assemble_row(
    base_row: dict[str, Any],
    kg_summary_stage1: str,
    kg_candidates_stage1: list[dict],
    oran_summary_stage2: str,
    oran_candidates_stage2: list[dict],
    shap_features: list[str],
) -> dict[str, Any]:
    """
    Assemble one v1.2 row by merging:
      - base row (authoritative 3GPP references, clause texts, causal mechanism)
      - Stage 1 KG evidence audit trail
      - Stage 2 O-RAN KG evidence audit trail
      - SHAP-ranked affected_kpis (if available)

    Authoritative fields from v1.0 are NEVER overwritten.
    """
    row = dict(base_row)  # copy; do not mutate base

    # SHAP update (v1.2 delta)
    if shap_features:
        row["affected_kpis"] = shap_features
        row["affected_kpis_source"] = "SHAP-ranked features from trained Random Forest classifier"
    else:
        # Preserve existing (empty list from v1.0)
        row.setdefault("affected_kpis", [])
        row["affected_kpis_source"] = "pending — SHAP update not yet applied"

    # KG evidence audit trail
    # Overwrite kg_evidence_summary only if Stage 1 found new evidence and
    # the base row does not already have a richer hand-curated summary.
    existing_kg_summary = row.get("kg_evidence_summary", "")
    if kg_candidates_stage1:
        # Append generated summary after the hand-curated one (or replace if empty)
        if existing_kg_summary:
            row["kg_evidence_summary"] = (
                existing_kg_summary
                + f" [Generated Stage-1 summary: {kg_summary_stage1}]"
            )
        else:
            row["kg_evidence_summary"] = kg_summary_stage1
    # Always record audit trails
    row["kg_stage1_candidates"] = kg_candidates_stage1
    row["kg_stage2_oran_candidates"] = oran_candidates_stage2
    row["kg_stage2_oran_summary"] = oran_summary_stage2

    # Validation status promotion
    # Promote to "kg_verified" if both KG stages found corroborating evidence
    # and the row does not have an outstanding pdf_action_required flag.
    current_status = row.get("validation_status", "pending")
    has_pdf_action = bool(row.get("pdf_action_required"))
    has_stage1 = bool(kg_candidates_stage1)
    has_stage2 = bool(oran_candidates_stage2)

    if current_status == "pending" and has_stage1 and has_stage2 and not has_pdf_action:
        row["validation_status"] = "kg_verified"
    elif current_status == "pending" and has_pdf_action:
        row["validation_status"] = "pending_pdf_confirmation"

    return row



# Main pipeline
def run_pipeline(
    kg_csv_dir: Path,
    oran_kg_path: Path,
    base_path: Path,
    shap_path: Path | None,
    output_path: Path,
    version_out: str = "alignment_table_v1.2",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run the full three-stage alignment table generation pipeline.

    Returns the assembled table dict.
    """
    # Load inputs 
    logger.info("Loading base table: %s", base_path)
    base = load_base_table(base_path)
    base_rows: list[dict] = base.get("rows", [])

    logger.info("Loading 3GPP KG CSVs from: %s", kg_csv_dir)
    kg_candidates_by_fault = load_kg_csv(kg_csv_dir)

    logger.info("Loading O-RAN KG nodes: %s", oran_kg_path)
    oran_nodes = load_oran_nodes(oran_kg_path)

    logger.info("Loading SHAP values: %s", shap_path or "(none)")
    shap_map = load_shap_values(shap_path)

    # Process each row 
    assembled_rows = []
    for base_row in base_rows:
        fault = base_row.get("telecomts_fault", "")
        logger.info("Processing row: %s", fault)

        # Stage 1
        raw_kg = kg_candidates_by_fault.get(fault, [])
        kg_summary, kg_audit = extract_kg_summary(fault, raw_kg)
        if raw_kg:
            logger.debug("  Stage 1: %d candidates → top %d", len(raw_kg), len(kg_audit))
        else:
            logger.warning("  Stage 1: no KG candidates found for '%s'", fault)

        # Stage 2
        oran_summary, oran_audit = score_oran_nodes(fault, oran_nodes)
        logger.debug("  Stage 2: %d O-RAN candidates", len(oran_audit))

        # Stage 3 — assemble
        shap_features = shap_map.get(fault, [])
        row = assemble_row(
            base_row,
            kg_summary, kg_audit,
            oran_summary, oran_audit,
            shap_features,
        )
        assembled_rows.append(row)

    # Build output document
    shap_applied = bool(shap_map)
    table = {
        "version": version_out,
        "generated_at": str(date.today()),
        "generated_by": "generate_alignment_table.py",
        "base_version": base.get("version", "alignment_table_v1.0"),
        "status": "DRAFT" if not shap_applied else "DRAFT_WITH_SHAP",
        "shap_applied": shap_applied,
        "pipeline_stages": {
            "stage1_3gpp_kg": str(kg_csv_dir),
            "stage2_oran_kg": str(oran_kg_path),
            "stage3_base_validation": str(base_path),
        },
        "schema_note": base.get("schema_note", ""),
        "shap_update_note": base.get("shap_update_note", ""),
        "classification_scheme": base.get("classification_scheme", {}),
        "relation_type_definitions": base.get("relation_type_definitions", {}),
        "rows": assembled_rows,
        "jamming_note": base.get("jamming_note", {}),
        "pending_validations": base.get("pending_validations", []),
    }

    # Write or print
    if dry_run:
        print(json.dumps(table, indent=2, ensure_ascii=False))
        logger.info("Dry-run: output printed to stdout, no file written.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)
        logger.info("Written: %s  (%d rows)", output_path, len(assembled_rows))

    return table



# CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "KPI-RAG Alignment Table Generator — three-stage pipeline "
            "(3GPP KG discovery → O-RAN KG discovery → authoritative assembly). "
            "Reads alignment_table_v1.0.json as the authoritative base and "
            "produces alignment_table_v1.2.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--kg-csv",
        type=Path,
        default=Path("data/3GPP_KG"),
        metavar="DIR",
        help=(
            "Directory containing 3GPP KG investigation CSVs produced by "
            "src/rag/ scripts (default: data/3GPP_KG)."
        ),
    )
    p.add_argument(
        "--oran-kg",
        type=Path,
        default=Path("data/oran/extracted/oran_relevant_nodes.jsonl"),
        metavar="FILE",
        help=(
            "Path to oran_relevant_nodes.jsonl produced by "
            "src/oran/extract_oran_kg.py (default: "
            "data/oran/extracted/oran_relevant_nodes.jsonl)."
        ),
    )
    p.add_argument(
        "--base",
        type=Path,
        default=Path("src/rca/alignment_table_v1.0.json"),
        metavar="FILE",
        help=(
            "Path to alignment_table_v1.0.json — the hand-validated base "
            "table (default: src/rca/alignment_table_v1.0.json)."
        ),
    )
    p.add_argument(
        "--shap",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Optional path to SHAP-ranked features JSON. When supplied, "
            "affected_kpis in each row is populated from it. "
            "Expected schema: {fault_type: [ranked_channel, ...]}."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("src/rca/alignment_table_v1.2.json"),
        metavar="FILE",
        help=(
            "Output path for alignment_table_v1.2.json "
            "(default: src/rca/alignment_table_v1.2.json)."
        ),
    )
    p.add_argument(
        "--version-out",
        default="alignment_table_v1.2",
        metavar="VERSION",
        help=(
            "Version string written into the output JSON's 'version' field "
            "(default: alignment_table_v1.2)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assembled table JSON to stdout instead of writing a file.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate inputs
    if not args.base.exists():
        logger.error("Base table not found: %s", args.base)
        sys.exit(1)
    if not args.kg_csv.exists():
        logger.warning(
            "KG CSV directory not found: %s — Stage 1 will have no evidence.", args.kg_csv
        )
    if not args.oran_kg.exists():
        logger.warning(
            "O-RAN KG file not found: %s — Stage 2 will have no evidence.", args.oran_kg
        )

    run_pipeline(
        kg_csv_dir=args.kg_csv,
        oran_kg_path=args.oran_kg,
        base_path=args.base,
        shap_path=args.shap,
        output_path=args.output,
        version_out=args.version_out,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        logger.info("Done — alignment_table_v1.2.json is ready.")
        logger.info(
            "Next step: run build_rca_evidence.py --table %s", args.output
        )
        if not args.shap:
            logger.info(
                "SHAP update pending: re-run with --shap <shap_ranked_features.json> "
                "after the classifier is trained to populate affected_kpis."
            )


if __name__ == "__main__":
    main()