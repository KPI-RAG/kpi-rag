import streamlit as st
import json, logging
from src.config_loader import load_config
from src.schema import ClassifierOutput, LLMExplanation, RetrievedTicket
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain
from dashboard.components.detection_panel import render_detection_panel
from dashboard.components.shap_panel import render_shap_panel
from dashboard.components.kpi_signal_panel import render_kpi_signal_panel
from dashboard.components.explanation_panel import render_explanation_panel
from dashboard.components.sources_panel import render_sources_panel

EXAMPLE_PAYLOAD = {
  "anomaly_type": "Antenna Failure",
  "confidence": 0.87,
  "shap_top3": [
    {"channel": "RSRP",    "shap_value": -0.42, "direction": "below_normal"},
    {"channel": "DL_BLER", "shap_value":  0.28, "direction": "above_normal"},
    {"channel": "DL_MCS",  "shap_value": -0.19, "direction": "below_normal"}
  ],
  "signal_statistics": {
    "RSRP":    {"mean": -105.0, "std": 3.2, "min": -112.0, "max": -98.0},
    "DL_BLER": {"mean": 0.35, "std": 0.08, "min": 0.21, "max": 0.51}
  }
}

st.set_page_config(
    page_title="KPI-RAG: 5G Fault Diagnosis",
    page_icon="📡",
    layout="wide"
)

st.title("📡 KPI-RAG: 5G Network Fault Diagnosis")
st.caption("Explainable root-cause analysis grounded in 3GPP standards")

@st.cache_resource
def get_cfg():
    return load_config()

cfg = get_cfg()

st.sidebar.header("Input")
input_method = st.sidebar.radio(
    "Input source",
    ["Upload JSON", "Use example"]
)

if input_method == "Upload JSON":
    uploaded = st.sidebar.file_uploader(
        "ClassifierOutput JSON", type="json")
    if uploaded:
        payload = ClassifierOutput(**json.load(uploaded))
    else:
        payload = None
else:
    payload = ClassifierOutput(**EXAMPLE_PAYLOAD)

st.sidebar.divider()
st.sidebar.caption("Model: all-MiniLM-L6-v2 | DB: ChromaDB")

if payload is None:
    st.info("👈 Upload a ClassifierOutput JSON or select example")
    st.stop()
    import sys; sys.exit(0)

col1, col2 = st.columns([1, 2])

with col1:
    anomaly_val = payload.anomaly_type.value if hasattr(payload.anomaly_type, "value") else payload.anomaly_type
    render_detection_panel(
        anomaly_val,
        payload.confidence
    )
    st.divider()
    render_shap_panel([s.model_dump() for s in payload.shap_top3])

with col2:
    render_kpi_signal_panel(
        {k: v.model_dump() for k, v in
         payload.signal_statistics.items()}
    )

st.divider()

with st.spinner("Retrieving similar incidents..."):
    collection = get_collection(cfg)
    tickets, low_conf = query_from_classifier_output(
        payload, collection, cfg)

with st.spinner("Generating explanation..."):
    alignment = load_alignment_table("configs/alignment_table.json")
    explanation = explain(payload, tickets, cfg, alignment)

col3, col4 = st.columns([2, 1])

with col3:
    render_explanation_panel(explanation, low_conf)

with col4:
    render_sources_panel(tickets, low_conf)
