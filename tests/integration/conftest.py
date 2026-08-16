import pytest
from src.config_loader import load_config
from src.schema import ClassifierOutput, AnomalyType


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def all_payloads():
    """One ClassifierOutput per fault type — all 11."""
    return {
        "Antenna Failure": {
            "anomaly_type": "Antenna Failure",
            "confidence": 0.87,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.42, "direction": "below_normal"},
                {"channel": "DL_BLER", "shap_value":  0.28, "direction": "above_normal"},
                {"channel": "DL_MCS",  "shap_value": -0.19, "direction": "below_normal"},
            ],
            "signal_statistics": {
                "RSRP":    {"mean": -108, "std": 4.2, "min": -115, "max": -98},
                "DL_BLER": {"mean": 0.38, "std": 0.09, "min": 0.22, "max": 0.54},
            },
        },
        "Buffer Overflow (Gradual Buildup)": {
            "anomaly_type": "Buffer Overflow (Gradual Buildup)",
            "confidence": 0.79,
            "shap_top3": [
                {"channel": "UL_BLER",     "shap_value":  0.51, "direction": "above_normal"},
                {"channel": "DL_PRB_UTIL", "shap_value":  0.38, "direction": "above_normal"},
                {"channel": "DL_BLER",     "shap_value":  0.22, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "UL_BLER":     {"mean": 0.44, "std": 0.11, "min": 0.29, "max": 0.63},
                "DL_PRB_UTIL": {"mean": 0.91, "std": 0.04, "min": 0.82, "max": 0.98},
            },
        },
        "Co-Channel Interference (Mild)": {
            "anomaly_type": "Co-Channel Interference (Mild)",
            "confidence": 0.74,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.31, "direction": "below_normal"},
                {"channel": "DL_SINR", "shap_value": -0.28, "direction": "below_normal"},
                {"channel": "DL_BLER", "shap_value":  0.19, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "RSRP":    {"mean": -102, "std": 5.1, "min": -112, "max": -91},
                "DL_SINR": {"mean": 4.2,  "std": 2.3, "min": 0.8,  "max": 8.9},
            },
        },
        "Co-Channel Interference (Severe)": {
            "anomaly_type": "Co-Channel Interference (Severe)",
            "confidence": 0.91,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.58, "direction": "below_normal"},
                {"channel": "DL_SINR", "shap_value": -0.44, "direction": "below_normal"},
                {"channel": "DL_BLER", "shap_value":  0.31, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "RSRP":    {"mean": -115, "std": 6.2, "min": -124, "max": -103},
                "DL_SINR": {"mean": -1.8, "std": 3.1, "min": -8.2, "max":  3.4},
            },
        },
        "Faulty RF Filters (Temporal)": {
            "anomaly_type": "Faulty RF Filters (Temporal)",
            "confidence": 0.82,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.39, "direction": "below_normal"},
                {"channel": "UL_SNR",  "shap_value": -0.27, "direction": "below_normal"},
                {"channel": "DL_BLER", "shap_value":  0.21, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "RSRP":   {"mean": -106, "std": 3.8, "min": -113, "max": -97},
                "UL_SNR": {"mean": 6.1,  "std": 2.9, "min": 1.8,  "max": 11.4},
            },
        },
        "High Network Congestion (Gradual Buildup)": {
            "anomaly_type": "High Network Congestion (Gradual Buildup)",
            "confidence": 0.85,
            "shap_top3": [
                {"channel": "DL_PRB_UTIL", "shap_value":  0.48, "direction": "above_normal"},
                {"channel": "UL_PRB_UTIL", "shap_value":  0.35, "direction": "above_normal"},
                {"channel": "DL_BLER",     "shap_value":  0.22, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "DL_PRB_UTIL": {"mean": 0.93, "std": 0.03, "min": 0.86, "max": 0.99},
                "UL_PRB_UTIL": {"mean": 0.88, "std": 0.05, "min": 0.78, "max": 0.96},
            },
        },
        "High Network Congestion (Sudden Spike)": {
            "anomaly_type": "High Network Congestion (Sudden Spike)",
            "confidence": 0.88,
            "shap_top3": [
                {"channel": "DL_PRB_UTIL", "shap_value":  0.61, "direction": "above_normal"},
                {"channel": "UL_BLER",     "shap_value":  0.44, "direction": "above_normal"},
                {"channel": "DL_BLER",     "shap_value":  0.38, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "DL_PRB_UTIL": {"mean": 0.97, "std": 0.02, "min": 0.93, "max": 1.00},
                "UL_BLER":     {"mean": 0.51, "std": 0.13, "min": 0.31, "max": 0.69},
            },
        },
        "Doppler Shift (Severe)": {
            "anomaly_type": "Doppler Shift (Severe)",
            "confidence": 0.77,
            "shap_top3": [
                {"channel": "RSRP",   "shap_value": -0.44, "direction": "below_normal"},
                {"channel": "DL_MCS", "shap_value": -0.33, "direction": "below_normal"},
                {"channel": "UL_MCS", "shap_value": -0.28, "direction": "below_normal"},
            ],
            "signal_statistics": {
                "RSRP":   {"mean": -104, "std": 7.3, "min": -118, "max": -89},
                "DL_MCS": {"mean": 6.8,  "std": 3.2, "min": 2.0,  "max": 14.0},
            },
        },
        "Faulty Handover Algorithm (Too Frequent)": {
            "anomaly_type": "Faulty Handover Algorithm (Too Frequent)",
            "confidence": 0.83,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.33, "direction": "below_normal"},
                {"channel": "DL_MCS",  "shap_value": -0.27, "direction": "below_normal"},
                {"channel": "UL_BLER", "shap_value":  0.21, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "RSRP":    {"mean": -99,  "std": 4.4, "min": -109, "max": -88},
                "UL_BLER": {"mean": 0.31, "std": 0.09, "min": 0.19, "max": 0.44},
            },
        },
        "Resource Allocation Bugs": {
            "anomaly_type": "Resource Allocation Bugs",
            "confidence": 0.71,
            "shap_top3": [
                {"channel": "DL_PRB_UTIL", "shap_value": -0.41, "direction": "below_normal"},
                {"channel": "UL_PRB_UTIL", "shap_value": -0.35, "direction": "below_normal"},
                {"channel": "DL_MCS",      "shap_value":  0.29, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "DL_PRB_UTIL": {"mean": 0.45, "std": 0.18, "min": 0.18, "max": 0.79},
                "DL_MCS":      {"mean": 9.1,  "std": 4.2,  "min": 2.0,  "max": 18.0},
            },
        },
        "Jamming": {
            "anomaly_type": "Jamming",
            "confidence": 0.93,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.71, "direction": "below_normal"},
                {"channel": "UL_SNR",  "shap_value": -0.55, "direction": "below_normal"},
                {"channel": "DL_BLER", "shap_value":  0.48, "direction": "above_normal"},
            ],
            "signal_statistics": {
                "RSRP":   {"mean": -121, "std": 8.1, "min": -134, "max": -108},
                "UL_SNR": {"mean": -3.2, "std": 4.1, "min": -11.8, "max": 2.9},
            },
        },
    }
