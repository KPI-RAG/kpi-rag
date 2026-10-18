import streamlit as st
import pandas as pd

def render_shap_panel(shap_top3: list[dict]) -> None:
    if not shap_top3:
        st.info("No SHAP data available")
        return

    st.subheader("Top KPI Contributors (SHAP)")

    df = pd.DataFrame(shap_top3)
    # Label by feature: several top features can share one channel (RSRP_max, RSRP_std, ...)
    names = df["feature"] if "feature" in df and df["feature"].astype(bool).all() else df["channel"]
    direction = df["feature_vs_normal"].str.replace("_normal_mean", " normal").str.replace("_", " ")
    df["label"] = names + " (" + direction + ")"
    df["color"] = df["feature_vs_normal"].apply(
        lambda d: "#FF4B4B" if "above" in d else "#4B8BFF"
    )

    # With horizontal=True, x is still the category axis and y the bar length.
    st.bar_chart(data=df, x="label", y="shap_value", color="color", horizontal=True)
