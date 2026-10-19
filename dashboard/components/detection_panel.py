import streamlit as st

def render_detection_panel(anomaly_type: str | None, confidence: float | None) -> None:
    if anomaly_type is None or anomaly_type == "Normal":
        st.success("✅ No anomaly detected")
        st.stop()
    else:
        st.error(f"🚨 Fault Detected: {anomaly_type}")
        if confidence is not None:
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.1%}"
            )
        # Phase 2 JSON: for an alarm this is Model 2's probability of the predicted type
        st.caption("Uncalibrated Model 2 probability of this fault type")

