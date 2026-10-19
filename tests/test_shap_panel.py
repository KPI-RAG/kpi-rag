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
        {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
        {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"}
    ]
    render_shap_panel(data)
    
    mock_info.assert_not_called()
    mock_sub.assert_called_once_with("Top KPI Contributors (SHAP)")
    mock_bar.assert_called_once()
    kwargs = mock_bar.call_args.kwargs
    assert (kwargs["x"], kwargs["y"]) == ("label", "shap_value")   # values as bar lengths, not as categories
    assert list(kwargs["data"]["label"]) == ["RSRP (below normal)", "DL_BLER (above normal)"]


@patch("dashboard.components.shap_panel.st.subheader")
@patch("dashboard.components.shap_panel.st.bar_chart")
def test_render_shap_panel_labels_features_of_one_channel(mock_bar, mock_sub):
    data = [
        {"channel": "RSRP", "feature": "RSRP_max", "shap_value": 1.09, "feature_vs_normal": "above_normal_mean"},
        {"channel": "RSRP", "feature": "RSRP_std", "shap_value": 0.87, "feature_vs_normal": "above_normal_mean"},
    ]
    render_shap_panel(data)
    assert list(mock_bar.call_args.kwargs["data"]["label"]) == ["RSRP_max (above normal)", "RSRP_std (above normal)"]
