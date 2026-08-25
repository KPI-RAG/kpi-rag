import streamlit as st
import pandas as pd

def render_shap_panel(shap_top3: list[dict]) -> None:
    if not shap_top3:
        st.info("No SHAP data available")
        return
        
    st.subheader("Top KPI Contributors (SHAP)")
    
    df = pd.DataFrame(shap_top3)
    df["channel_label"] = df["channel"] + " (" + df["direction"] + ")"
    df["color"] = df["direction"].apply(lambda d: "#FF4B4B" if d == "above_normal" else "#4B8BFF")
    
    try:
        st.bar_chart(
            data=df,
            x="shap_value",
            y="channel_label",
            color="color",
            horizontal=True
        )
    except TypeError:
        # Fallback if horizontal is not supported
        st.bar_chart(
            data=df,
            x="channel_label",
            y="shap_value",
            color="color"
        )
