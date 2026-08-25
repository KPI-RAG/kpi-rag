import pytest
from unittest.mock import patch, MagicMock
from dashboard.components.shap_panel import render_shap_panel

@patch("dashboard.components.shap_panel.st.info")
@patch("dashboard.components.shap_panel.st.subheader")
@patch("dashboard.components.shap_panel.st.bar_chart")
def test_render_shap_panel_empty(mock_bar, mock_sub, mock_info):
    render_shap_panel([])
    mock_info.assert_called_once_with("No SHAP data available")
    mock_sub.assert_not_called()
    mock_bar.assert_not_called()

@patch("dashboard.components.shap_panel.st.info")
@patch("dashboard.components.shap_panel.st.subheader")
@patch("dashboard.components.shap_panel.st.bar_chart")
def test_render_shap_panel_valid(mock_bar, mock_sub, mock_info):
    data = [
        {"channel": "RSRP", "shap_value": -0.42, "direction": "below_normal"},
        {"channel": "DL_BLER", "shap_value": 0.28, "direction": "above_normal"}
    ]
    render_shap_panel(data)
    
    mock_info.assert_not_called()
    mock_sub.assert_called_once_with("Top KPI Contributors (SHAP)")
    mock_bar.assert_called_once()
