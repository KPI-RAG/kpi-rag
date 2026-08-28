import streamlit as st
import pandas as pd

def render_kpi_signal_panel(signal_statistics: dict[str, float], window_length: int = 128) -> None:
    if not signal_statistics:
        st.info("No signal data available")
        return
        
    st.subheader("KPI Signal Window (128 timesteps @ 10ms)")
    
    # signal_statistics is now flat: {"RSRP_mean": -77.6, "RSRP_std": 0.78, ...}
    # Collect unique channel names by stripping trailing _mean/_std/_min/_max
    channels = set()
    for key in signal_statistics:
        for suffix in ("_mean", "_std", "_min", "_max"):
            if key.endswith(suffix):
                channels.add(key[: -len(suffix)])
                break

    data = {}
    for channel in sorted(channels):
        mean = signal_statistics.get(f"{channel}_mean", 0.0)
        std = signal_statistics.get(f"{channel}_std", 0.0)
        
        data[f"{channel} (mean)"] = [mean] * window_length
        data[f"{channel} (upper)"] = [mean + std] * window_length
        data[f"{channel} (lower)"] = [mean - std] * window_length
        
    df = pd.DataFrame(data)
    st.line_chart(df)
