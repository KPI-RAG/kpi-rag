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
                {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
                {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -108, "RSRP_std": 4.2, "RSRP_min": -115, "RSRP_max": -98,
                "DL_BLER_mean": 0.38, "DL_BLER_std": 0.09, "DL_BLER_min": 0.22, "DL_BLER_max": 0.54,
            },
        },
        "Buffer Overflow (Gradual Buildup)": {
            "anomaly_type": "Buffer Overflow (Gradual Buildup)",
            "confidence": 0.79,
            "shap_top3": [
                {"channel": "UL_BLER",     "shap_value":  0.51, "feature_vs_normal": "above_normal_mean"},
                {"channel": "DL_PRB_UTIL", "shap_value":  0.38, "feature_vs_normal": "above_normal_mean"},
                {"channel": "DL_BLER",     "shap_value":  0.22, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "UL_BLER_mean": 0.44, "UL_BLER_std": 0.11, "UL_BLER_min": 0.29, "UL_BLER_max": 0.63,
                "DL_PRB_UTIL_mean": 0.91, "DL_PRB_UTIL_std": 0.04, "DL_PRB_UTIL_min": 0.82, "DL_PRB_UTIL_max": 0.98,
            },
        },
        "Co-Channel Interference (Mild)": {
            "anomaly_type": "Co-Channel Interference (Mild)",
            "confidence": 0.74,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.31, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_SINR", "shap_value": -0.28, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_BLER", "shap_value":  0.19, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -102, "RSRP_std": 5.1, "RSRP_min": -112, "RSRP_max": -91,
                "DL_SINR_mean": 4.2,  "DL_SINR_std": 2.3, "DL_SINR_min": 0.8,  "DL_SINR_max": 8.9,
            },
        },
        "Co-Channel Interference (Severe)": {
            "anomaly_type": "Co-Channel Interference (Severe)",
            "confidence": 0.91,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.58, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_SINR", "shap_value": -0.44, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_BLER", "shap_value":  0.31, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -115, "RSRP_std": 6.2, "RSRP_min": -124, "RSRP_max": -103,
                "DL_SINR_mean": -1.8, "DL_SINR_std": 3.1, "DL_SINR_min": -8.2, "DL_SINR_max": 3.4,
            },
        },
        "Faulty RF Filters (Temporal)": {
            "anomaly_type": "Faulty RF Filters (Temporal)",
            "confidence": 0.82,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.39, "feature_vs_normal": "below_normal_mean"},
                {"channel": "UL_SNR",  "shap_value": -0.27, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_BLER", "shap_value":  0.21, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -106, "RSRP_std": 3.8, "RSRP_min": -113, "RSRP_max": -97,
                "UL_SNR_mean": 6.1,  "UL_SNR_std": 2.9, "UL_SNR_min": 1.8,  "UL_SNR_max": 11.4,
            },
        },
        "High Network Congestion (Gradual Buildup)": {
            "anomaly_type": "High Network Congestion (Gradual Buildup)",
            "confidence": 0.85,
            "shap_top3": [
                {"channel": "DL_PRB_UTIL", "shap_value":  0.48, "feature_vs_normal": "above_normal_mean"},
                {"channel": "UL_PRB_UTIL", "shap_value":  0.35, "feature_vs_normal": "above_normal_mean"},
                {"channel": "DL_BLER",     "shap_value":  0.22, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "DL_PRB_UTIL_mean": 0.93, "DL_PRB_UTIL_std": 0.03, "DL_PRB_UTIL_min": 0.86, "DL_PRB_UTIL_max": 0.99,
                "UL_PRB_UTIL_mean": 0.88, "UL_PRB_UTIL_std": 0.05, "UL_PRB_UTIL_min": 0.78, "UL_PRB_UTIL_max": 0.96,
            },
        },
        "High Network Congestion (Sudden Spike)": {
            "anomaly_type": "High Network Congestion (Sudden Spike)",
            "confidence": 0.88,
            "shap_top3": [
                {"channel": "DL_PRB_UTIL", "shap_value":  0.61, "feature_vs_normal": "above_normal_mean"},
                {"channel": "UL_BLER",     "shap_value":  0.44, "feature_vs_normal": "above_normal_mean"},
                {"channel": "DL_BLER",     "shap_value":  0.38, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "DL_PRB_UTIL_mean": 0.97, "DL_PRB_UTIL_std": 0.02, "DL_PRB_UTIL_min": 0.93, "DL_PRB_UTIL_max": 1.00,
                "UL_BLER_mean": 0.51,     "UL_BLER_std": 0.13,     "UL_BLER_min": 0.31,     "UL_BLER_max": 0.69,
            },
        },
        "Doppler Shift (Severe)": {
            "anomaly_type": "Doppler Shift (Severe)",
            "confidence": 0.77,
            "shap_top3": [
                {"channel": "RSRP",   "shap_value": -0.44, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_MCS", "shap_value": -0.33, "feature_vs_normal": "below_normal_mean"},
                {"channel": "UL_MCS", "shap_value": -0.28, "feature_vs_normal": "below_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -104, "RSRP_std": 7.3, "RSRP_min": -118, "RSRP_max": -89,
                "DL_MCS_mean": 6.8,  "DL_MCS_std": 3.2, "DL_MCS_min": 2.0,  "DL_MCS_max": 14.0,
            },
        },
        "Faulty Handover Algorithm (Too Frequent)": {
            "anomaly_type": "Faulty Handover Algorithm (Too Frequent)",
            "confidence": 0.83,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.33, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_MCS",  "shap_value": -0.27, "feature_vs_normal": "below_normal_mean"},
                {"channel": "UL_BLER", "shap_value":  0.21, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -99, "RSRP_std": 4.4, "RSRP_min": -109, "RSRP_max": -88,
                "UL_BLER_mean": 0.31, "UL_BLER_std": 0.09, "UL_BLER_min": 0.19, "UL_BLER_max": 0.44,
            },
        },
        "Resource Allocation Bugs": {
            "anomaly_type": "Resource Allocation Bugs",
            "confidence": 0.71,
            "shap_top3": [
                {"channel": "DL_PRB_UTIL", "shap_value": -0.41, "feature_vs_normal": "below_normal_mean"},
                {"channel": "UL_PRB_UTIL", "shap_value": -0.35, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_MCS",      "shap_value":  0.29, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "DL_PRB_UTIL_mean": 0.45, "DL_PRB_UTIL_std": 0.18, "DL_PRB_UTIL_min": 0.18, "DL_PRB_UTIL_max": 0.79,
                "DL_MCS_mean": 9.1,       "DL_MCS_std": 4.2,       "DL_MCS_min": 2.0,       "DL_MCS_max": 18.0,
            },
        },
        "Jamming": {
            "anomaly_type": "Jamming",
            "confidence": 0.93,
            "shap_top3": [
                {"channel": "RSRP",    "shap_value": -0.71, "feature_vs_normal": "below_normal_mean"},
                {"channel": "UL_SNR",  "shap_value": -0.55, "feature_vs_normal": "below_normal_mean"},
                {"channel": "DL_BLER", "shap_value":  0.48, "feature_vs_normal": "above_normal_mean"},
            ],
            "signal_statistics": {
                "RSRP_mean": -121, "RSRP_std": 8.1, "RSRP_min": -134, "RSRP_max": -108,
                "UL_SNR_mean": -3.2, "UL_SNR_std": 4.1, "UL_SNR_min": -11.8, "UL_SNR_max": 2.9,
            },
        },
    }
