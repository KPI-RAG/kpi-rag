"""RCA Evidence panel -- shows Rodina layer A/B/C RCA evidence."""
import streamlit as st


def render_rca_panel(rca_record: dict):
    """Display 3-layer RCA evidence from rca_evidence.json (Rodina, P3).

    Args:
        rca_record: One record from rca_evidence.json, or None / empty dict.
    """
    if not rca_record:
        st.info("No RCA evidence available for this window.")
        return

    st.subheader("🔍 Root Cause Analysis Evidence")

    # Layer C -- Standards grounding (most important for citation)
    layer_c = rca_record.get("layer_c_domain_standards", {})
    if layer_c:
        with st.expander("📋 Standards Grounding (Layer C)", expanded=True):
            ref = layer_c.get("3gpp_reference", "N/A")
            oran = layer_c.get("oran_component", "N/A")
            mechanism = layer_c.get("causal_mechanism", "N/A")
            st.markdown(f"**3GPP Reference:** `{ref}`")
            st.markdown(f"**O-RAN Component:** {oran}")
            st.markdown(f"**Causal Mechanism:** {mechanism}")

    # Layer B -- SHAP model attribution
    layer_b = rca_record.get("layer_b_model_attribution", [])
    if layer_b:
        with st.expander("📊 Model Attribution (Layer B -- SHAP)"):
            for feat in layer_b:
                direction = (
                    "↑" if "above" in feat.get("feature_vs_normal", "") else "↓"
                )
                shap_val = feat.get("shap_value", 0)
                st.markdown(
                    f"**{feat.get('feature', '')}** {direction} "
                    f"SHAP={shap_val:.3f}"
                )

    # Coverage summary
    summary = rca_record.get("coverage_summary", {})
    if summary:
        with st.expander("📈 KPI Evidence Coverage"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Strong Evidence", summary.get("strong_count", 0))
            col2.metric("Supporting", summary.get("supporting_count", 0))
            col3.metric("Missing", summary.get("missing_count", 0))

    # Pipeline note for Jamming / fallback cases
    fallback = rca_record.get("pipeline_fallback", "")
    if fallback:
        st.caption(f"ℹ️ {fallback}")
