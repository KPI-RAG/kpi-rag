import pytest
from unittest.mock import patch
from dashboard.components.kpi_signal_panel import render_kpi_signal_panel

@patch("dashboard.components.kpi_signal_panel.st.info")
@patch("dashboard.components.kpi_signal_panel.st.subheader")
@patch("dashboard.components.kpi_signal_panel.st.line_chart")
def test_render_kpi_signal_panel_empty(mock_line, mock_sub, mock_info):
    render_kpi_signal_panel({})
    mock_info.assert_called_once_with("No signal data available")
    mock_sub.assert_not_called()
    mock_line.assert_not_called()

@patch("dashboard.components.kpi_signal_panel.st.info")
@patch("dashboard.components.kpi_signal_panel.st.subheader")
@patch("dashboard.components.kpi_signal_panel.st.line_chart")
def test_render_kpi_signal_panel_valid(mock_line, mock_sub, mock_info):
    # signal_statistics is now flat
    stats = {
        "RSRP_mean": -105.0, "RSRP_std": 3.2, "RSRP_min": -112.0, "RSRP_max": -98.0
    }
    render_kpi_signal_panel(stats)
    
    mock_info.assert_not_called()
    mock_sub.assert_called_once_with("KPI Signal Window (128 timesteps @ 10ms)")
    mock_line.assert_called_once()
    
    df_called = mock_line.call_args[0][0]
    assert df_called.shape == (128, 3)
    assert df_called.iloc[0]["RSRP (mean)"] == -105.0
