import sys
import os
import json
import logging

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# On Streamlit Cloud, secrets come from environment directly.
# load_dotenv() is only needed for local development.
if os.path.exists(os.path.join(REPO_ROOT, ".env")):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

import streamlit as st

st.set_page_config(
    page_title="KPI-RAG: 5G Fault Diagnosis",
    page_icon="📡",
    layout="wide"
)

from src.config_loader import load_config

# Check the configured backend's key before loading anything heavy (ollama needs none).
_key_name = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}.get(
    load_config(os.path.join(REPO_ROOT, "configs", "config.yaml"))["llm"]["backend"]
)
if _key_name and not os.environ.get(_key_name):
    st.error(
        f"⚠️ **{_key_name} not configured.**  \n"
        "Add it in **App Settings → Secrets** on Streamlit Community Cloud, "
        "or set it in your local `.env` file."
    )
    st.stop()

from src.schema import ClassifierOutput, SHAPEntry
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain, alignment_path
from src.rca_loader import RCALoader
from dashboard.components.detection_panel import render_detection_panel
from dashboard.components.shap_panel import render_shap_panel
from dashboard.components.kpi_signal_panel import render_kpi_signal_panel
from dashboard.components.explanation_panel import render_explanation_panel
from dashboard.components.sources_panel import render_sources_panel
from dashboard.components.rca_panel import render_rca_panel

logger = logging.getLogger(__name__)

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
    cfg = load_config()
    chroma_path = cfg.get("rag", {}).get("chroma_db_path", "data/chroma_db")
    if not os.path.isabs(chroma_path):
        cfg["rag"]["chroma_db_path"] = os.path.join(REPO_ROOT, chroma_path)
    return cfg


@st.cache_resource(show_spinner="Loading ChromaDB collection…")
def load_chromadb():
    """Load ChromaDB collection once and cache for the entire app lifetime.
    Critical for 1GB RAM limit — avoids reloading on every user interaction.
    """
    cfg = get_cfg()
    return get_collection(cfg)


@st.cache_data(show_spinner=False)
def load_alignment():
    """Load alignment table once from disk and cache."""
    return load_alignment_table(os.path.join(REPO_ROOT, alignment_path(get_cfg())))


@st.cache_resource(show_spinner=False)
def load_rca():
    """Load the RCALoader once; returns None if rca_evidence.json is absent."""
    rca_path = os.path.join(REPO_ROOT, "data", "processed", "rca_evidence.json")
    if os.path.exists(rca_path):
        return RCALoader(rca_path)
    return None



@st.cache_data(show_spinner=False)
def load_layer2_windows() -> list:
    """Windows Phase 2 flagged as faults, as ClassifierOutput objects.

    Every record in the Phase 2 file is a test-split window, so none of them has
    its own ticket in the (train-only) ChromaDB index.
    """
    data_cfg = get_cfg()["data"]
    path = os.path.join(REPO_ROOT, data_cfg["processed_path"], data_cfg["layer2_output_file"])
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    windows = []
    for r in records:
        if r["anomaly_type"] == "Normal":
            continue
        try:
            windows.append({"window_index": r["window_index"], "payload": ClassifierOutput(**r)})
        except Exception as e:
            logger.warning("Skipping malformed layer-2 window record: %s", e)
    return windows


cfg = get_cfg()
rca_loader = load_rca()
layer2_windows = load_layer2_windows()

st.title("📡 KPI-RAG: 5G Network Fault Diagnosis")
st.caption("Explainable root-cause analysis grounded in 3GPP standards")

st.sidebar.header("Input")
input_method = st.sidebar.radio(
    "Input source",
    ["Upload JSON", "Use example", "Browse Dataset Windows"],
    index=1
)

window_index = None

if input_method == "Upload JSON":
    uploaded = st.sidebar.file_uploader("ClassifierOutput JSON", type="json")
    if uploaded:
        try:
            payload = ClassifierOutput(**json.load(uploaded))
        except Exception as e:   # bad JSON or schema mismatch
            st.error(f"Not a valid ClassifierOutput JSON: {e}")
            st.stop()
        active_key = "uploaded"
    else:
        payload = None
        active_key = "none"

elif input_method == "Browse Dataset Windows":
    if not layer2_windows:
        st.warning(f"{cfg['data']['layer2_output_file']} not found — check data/processed/")
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
    active_key = f"window_{window_index}_{payload.anomaly_type.value}"
    st.sidebar.caption(
        f"window_index={window_index} | fault={payload.anomaly_type.value} | "
        f"conf={payload.confidence:.0%}"
    )

else:  # Use example
    payload = ClassifierOutput(**EXAMPLE_PAYLOAD)
    active_key = "example_antenna_failure"

st.sidebar.divider()
_llm = cfg["llm"]
st.sidebar.caption(
    f"Model: {cfg['rag']['embedding_model']} | DB: ChromaDB | "
    f"LLM: {_llm.get(_llm['backend'] + '_model', _llm['backend'])}"
)

if payload is None:
    st.info("👈 Upload a ClassifierOutput JSON or select example")
    st.stop()

# RCA evidence is only available when a dataset window is selected, and is only
# injected when the record is for the same fault the payload names.
rca_context = ""
if (rca_loader is not None and window_index is not None
        and (rca_loader.get(window_index) or {}).get("predicted_fault") == payload.anomaly_type.value):
    rca_context = rca_loader.get_prompt_context(window_index)
if rca_context:
    st.sidebar.caption("✅ RCA evidence injected into prompt")
else:
    st.sidebar.caption("ℹ️ No RCA context for this input")

if payload is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        anomaly_val = payload.anomaly_type.value if hasattr(payload.anomaly_type, "value") else payload.anomaly_type
        render_detection_panel(anomaly_val, payload.confidence)
        st.divider()
        render_shap_panel([s.model_dump() for s in payload.shap_top3])

    with col2:
        render_kpi_signal_panel(
            payload.signal_statistics,
            window_length=cfg.get("data", {}).get("window_length", 128)
        )

st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    generate_clicked = st.button("🔍 Generate Explanation", type="primary", use_container_width=True)

if generate_clicked:
    with st.spinner("Retrieving similar incidents..."):
        collection = load_chromadb()
        tickets, low_conf = query_from_classifier_output(payload, collection, cfg)

    with st.spinner("Generating explanation..."):
        alignment = load_alignment()
        explanation = explain(payload, tickets, cfg, alignment, rca_context=rca_context)

    st.session_state["saved_explanation"] = explanation
    st.session_state["saved_tickets"] = tickets
    st.session_state["saved_low_conf"] = low_conf
    st.session_state["saved_key"] = active_key

if st.session_state.get("saved_key") == active_key and st.session_state.get("saved_explanation") is not None:
    explanation = st.session_state["saved_explanation"]
    tickets = st.session_state["saved_tickets"]
    low_conf = st.session_state["saved_low_conf"]

    col3, col4 = st.columns([2, 1])

    with col3:
        render_explanation_panel(explanation, low_conf)

    with col4:
        render_sources_panel(tickets, low_conf)
else:
    st.info("👆 Click **Generate Explanation** to retrieve 3GPP standards and generate diagnosis.")

st.divider()

# RCA Evidence panel. rca_evidence.json covers the 289 non-Jamming Phase 2 alarms;
# for other inputs show a representative record and say so.
rca_record = rca_loader.get(window_index) if (rca_loader and window_index is not None) else None
if rca_record is None and rca_loader is not None:
    fault_val = payload.anomaly_type.value
    rca_record = rca_loader.find_by_fault(fault_val)
    if rca_record is not None:
        st.caption(
            f"ℹ️ No RCA evidence exists for this exact window — showing a representative "
            f"{fault_val} record (window {rca_record['window_index']}). It was not sent to the LLM."
        )
render_rca_panel(rca_record)

