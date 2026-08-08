import streamlit as st
import pandas as pd

def render_kpi_signal_panel(signal_statistics: dict[str, dict], window_length: int = 128) -> None:
    if not signal_statistics:
        st.info("No signal data available")
        return
        
    st.subheader("KPI Signal Window (128 timesteps @ 100ms)")
    
    data = {}
    for channel, stats in signal_statistics.items():
        mean = stats["mean"]
        std = stats["std"]
        
        data[f"{channel} (mean)"] = [mean] * window_length
        data[f"{channel} (upper)"] = [mean + std] * window_length
        data[f"{channel} (lower)"] = [mean - std] * window_length
        
    df = pd.DataFrame(data)
    st.line_chart(df)
