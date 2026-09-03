import sys
import os
import json
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.config_loader import load_config
from src.schema import ClassifierOutput, SHAPEntry
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain
from dashboard.components.detection_panel import render_detection_panel
from dashboard.components.shap_panel import render_shap_panel
from dashboard.components.kpi_signal_panel import render_kpi_signal_panel
from dashboard.components.explanation_panel import render_explanation_panel
from dashboard.components.sources_panel import render_sources_panel
from dashboard.components.rca_panel import render_rca_panel

EXAMPLE_PAYLOAD = {
  "anomaly_type": "Antenna Failure",
  "confidence": 0.87,
  "shap_top3": [
    {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
    {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
    {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"}
  ],
  "signal_statistics": {
    "RSRP_mean": -105.0, "RSRP_std": 3.2, "RSRP_min": -112.0, "RSRP_max": -98.0,
    "DL_BLER_mean": 0.35, "DL_BLER_std": 0.08, "DL_BLER_min": 0.21, "DL_BLER_max": 0.51,
  }
}


@st.cache_resource
def get_cfg():
    return load_config()


@st.cache_data(show_spinner=False)
def load_rca_evidence(path: str = "data/processed/rca_evidence.json") -> dict:
    """Load rca_evidence.json and index by window_index (Rodina, P3)."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {r["window_index"]: r for r in records}


@st.cache_data(show_spinner=False)
def load_layer2_windows(path: str = "data/processed/layer2_rag_handoff_sessionsplit.json") -> list:
    """Load Raneem layer2 handoff windows as ClassifierOutput objects (P2)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    windows = []
    for r in records:
        try:
            payload = ClassifierOutput(
                anomaly_type=r["predicted_fault_type"],
                confidence=r["confidence"],
                shap_top3=r["shap_top3"],
                signal_statistics=r["signal_statistics"],
            )
            windows.append({
                "window_index": r.get("window_index", len(windows)),
                "payload": payload,
            })
        except Exception:
            pass
    return windows


cfg = get_cfg()
rca_evidence = load_rca_evidence()
layer2_windows = load_layer2_windows()

st.set_page_config(
    page_title="KPI-RAG: 5G Fault Diagnosis",
    page_icon="📡",
    layout="wide"
)

st.title("📡 KPI-RAG: 5G Network Fault Diagnosis")
st.caption("Explainable root-cause analysis grounded in 3GPP standards")

st.sidebar.header("Input")
input_method = st.sidebar.radio(
    "Input source",
    ["Upload JSON", "Use example", "Browse Raneem windows"]
)

window_index = None

if input_method == "Upload JSON":
    uploaded = st.sidebar.file_uploader("ClassifierOutput JSON", type="json")
    if uploaded:
        payload = ClassifierOutput(**json.load(uploaded))
    else:
        payload = None

elif input_method == "Browse Raneem windows":
    if not layer2_windows:
        st.warning("layer2_rag_handoff_sessionsplit.json not found — check data/processed/")
        st.stop()
    fault_types = sorted(set(w["payload"].anomaly_type.value for w in layer2_windows))
    selected_fault = st.sidebar.selectbox("Filter by fault type", ["All"] + fault_types)
    filtered = (
        layer2_windows if selected_fault == "All"
        else [w for w in layer2_windows if w["payload"].anomaly_type.value == selected_fault]
    )
    if not filtered:
        st.info("No windows match filter.")
        st.stop()
    idx_label = st.sidebar.slider("Window", 0, len(filtered) - 1, 0)
    selected = filtered[idx_label]
    window_index = selected["window_index"]
    payload = selected["payload"]
    st.sidebar.caption(
        f"window_index={window_index} | fault={payload.anomaly_type.value} | "
        f"conf={payload.confidence:.0%}"
    )

else:  # Use example
    payload = ClassifierOutput(**EXAMPLE_PAYLOAD)

st.sidebar.divider()
st.sidebar.caption("Model: all-MiniLM-L6-v2 | DB: ChromaDB | LLM: Gemini 3.5 Flash Lite")

if payload is None:
    st.info("👈 Upload a ClassifierOutput JSON or select example")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    anomaly_val = payload.anomaly_type.value if hasattr(payload.anomaly_type, "value") else payload.anomaly_type
    render_detection_panel(anomaly_val, payload.confidence)
    st.divider()
    render_shap_panel([s.model_dump() for s in payload.shap_top3])

with col2:
    render_kpi_signal_panel(payload.signal_statistics)


st.divider()

with st.spinner("Retrieving similar incidents..."):
    collection = get_collection(cfg)
    tickets, low_conf = query_from_classifier_output(payload, collection, cfg)

with st.spinner("Generating explanation..."):
    alignment = load_alignment_table("configs/alignment_table.json")
    explanation = explain(payload, tickets, cfg, alignment)

col3, col4 = st.columns([2, 1])

with col3:
    render_explanation_panel(explanation, low_conf)

with col4:
    render_sources_panel(tickets, low_conf)

st.divider()

# RCA Evidence panel (Rodina, P3)
rca_record = rca_evidence.get(window_index) if window_index is not None else None
if rca_record is None and rca_evidence:
    # Fallback: match by fault type for example/upload modes
    fault_val = payload.anomaly_type.value if hasattr(payload.anomaly_type, "value") else str(payload.anomaly_type)
    for rec in rca_evidence.values():
        if rec.get("predicted_fault") == fault_val:
            rca_record = rec
            break
render_rca_panel(rca_record)
