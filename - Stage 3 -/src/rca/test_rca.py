"""
test_rca.py
-----------
Smoke test for the RCA layer using synthetic Layer 2 examples.
Covers: normal fault, Jamming fallback, unknown fault.

Run from the rca/ directory:
    python test_rca.py
"""

import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s %(message)s")

# Add parent dir so rca_utils and build_rca_evidence are importable
sys.path.insert(0, str(Path(__file__).parent))

from rca_utils import load_alignment_table
from build_rca_evidence import build_rca_for_window

TABLE_PATH = Path(__file__).parent.parent / "alignment_table_v1_2.json"

# ── Synthetic Layer 2 windows ─────────────────────────────────────────────────

WINDOWS = [
    # 1. Co-Channel Interference Severe — should produce alignment_table source
    {
        "window_index": 0,
        "ground_truth_anomaly_type": "Co-Channel Interference (Severe)",
        "predicted_fault_type": "Co-Channel Interference (Severe)",
        "confidence": 0.91,
        "shap_top3": [
            {"feature": "UL_SNR",  "shap_value": -0.84, "feature_vs_normal": "below_normal_mean"},
            {"feature": "UL_BLER", "shap_value":  0.71, "feature_vs_normal": "above_normal_mean"},
            {"feature": "DL_MCS",  "shap_value": -0.53, "feature_vs_normal": "below_normal_mean"},
        ],
        "signal_statistics": {
            "UL_SNR_mean":  20.72, "UL_SNR_std":  3.1,
            "UL_BLER_mean": 0.089, "UL_BLER_std": 0.02,
            "DL_MCS_mean":  12.4,  "DL_MCS_std":  2.3,
            "RSRP_mean":   -88.5,  "RSRP_std":    5.0,
            "DL_BLER_mean": 0.05,  "DL_BLER_std": 0.01,
            "PRB_Utilization_DL_mean": 0.42,
            "PRB_Utilization_UL_mean": 0.38,
        },
    },
    # 2. Jamming — should produce jamming_fallback source
    {
        "window_index": 1,
        "ground_truth_anomaly_type": "Jamming",
        "predicted_fault_type": "Jamming",
        "confidence": 0.989,
        "shap_top3": [
            {"feature": "RSRP_max",  "shap_value": 2.09, "feature_vs_normal": "above_normal_mean"},
            {"feature": "RSRP_mean", "shap_value": 1.87, "feature_vs_normal": "above_normal_mean"},
            {"feature": "RSRP_min",  "shap_value": 1.44, "feature_vs_normal": "above_normal_mean"},
        ],
        "signal_statistics": {
            "RSRP_mean": -65.0, "RSRP_std": 4.2, "RSRP_min": -72.0, "RSRP_max": -58.0,
            "UL_SNR_mean": 8.3, "UL_BLER_mean": 0.31,
        },
    },
    # 3. High Network Congestion Static
    {
        "window_index": 2,
        "ground_truth_anomaly_type": "High Network Congestion (Static)",
        "predicted_fault_type": "High Network Congestion (Static)",
        "confidence": 0.87,
        "shap_top3": [
            {"feature": "PRB_Utilization_DL", "shap_value":  1.23, "feature_vs_normal": "above_normal_mean"},
            {"feature": "PRB_Utilization_UL", "shap_value":  1.10, "feature_vs_normal": "above_normal_mean"},
            {"feature": "TX_Bytes",           "shap_value": -0.92, "feature_vs_normal": "below_normal_mean"},
        ],
        "signal_statistics": {
            "PRB_Utilization_DL_mean": 0.95, "PRB_Utilization_DL_std": 0.03,
            "PRB_Utilization_UL_mean": 0.93, "PRB_Utilization_UL_std": 0.04,
            "TX_Bytes_mean": 1200.0, "TX_Bytes_std": 300.0,
            "Estimated_UL_Buffer_mean": 88400.0,
            "DL_MCS_mean": 8.1, "UL_MCS_mean": 7.5,
        },
    },
    # 4. Unknown fault — tests graceful handling
    {
        "window_index": 3,
        "ground_truth_anomaly_type": "???",
        "predicted_fault_type": "Some Unknown Fault Type",
        "confidence": 0.55,
        "shap_top3": [
            {"feature": "RSRP_mean", "shap_value": 0.5, "feature_vs_normal": "above_normal_mean"},
        ],
        "signal_statistics": {
            "RSRP_mean": -90.0,
        },
    },
]


def pretty(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    if not TABLE_PATH.exists():
        print(f"ERROR: alignment table not found at {TABLE_PATH}")
        sys.exit(1)

    table = load_alignment_table(TABLE_PATH)

    for window in WINDOWS:
        rca = build_rca_for_window(window, table)
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"Window {rca['window_index']}  |  predicted: {rca['predicted_fault']}")
        print(f"  rca_source:       {rca['rca_source']}")
        print(f"  confidence:       {rca['confidence']}")
        print(f"  coverage_summary: {rca['coverage_summary']}")
        print()
        print("  KPI evidence:")
        for e in rca.get("kpi_evidence", []):
            shap = f"shap={e['shap_value']:+.3f}" if e.get("shap_value") is not None else "shap=—"
            obs  = f"obs_mean={e['observed_mean']:.3f}" if e.get("observed_mean") is not None else "obs_mean=—"
            print(
                f"    [{e['evidence_status']:12s}]  {e['kpi']:30s}  "
                f"{shap:15s}  {obs}"
            )
        print()
        if rca.get("standards_evidence"):
            se = rca["standards_evidence"]
            print(f"  3GPP reference:   {se['3gpp_reference']}")
            print(f"  O-RAN component:  {se['oran_component']}")
            print(f"  relation_type:    {se['relation_type']}")
            print(f"  support_class:    {se['support_class']}")
            print(f"  evidence_strength:{se['evidence_strength']}")
        else:
            print("  standards_evidence: None (Jamming or unknown fault)")

    print(f"\n{'=' * 70}")
    print("All test windows processed successfully.")


if __name__ == "__main__":
    main()
