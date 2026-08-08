import streamlit as st
from src.schema import LLMExplanation

def render_explanation_panel(explanation: LLMExplanation | None, low_confidence: bool = False) -> None:
    if low_confidence:
        st.warning("⚠️ Low historical precedent — no similar incidents found in training data")
        
    if explanation is None:
        st.info("No explanation generated")
        return
        
    st.subheader("Root Cause")
    st.write(explanation.root_cause)
    
    st.subheader("3GPP Reference")
    if explanation.reference_valid:
        st.success(f"✅ {explanation.gpp_reference}")
    else:
        st.error(f"❌ {explanation.gpp_reference} (unvalidated)")
        
    st.subheader("O-RAN Component")
    st.write(explanation.oran_component)
    
    st.subheader("Recommended Action")
    st.write(explanation.recommended_action)
    
    st.subheader("Reasoning")
    st.write(explanation.reasoning_trace)
    
    if explanation.template_generated:
        st.caption("📋 Generated from alignment table template (LLM parse failed)")
