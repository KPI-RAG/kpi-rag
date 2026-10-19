import pytest
from unittest.mock import patch
from dashboard.components.detection_panel import render_detection_panel

@patch("dashboard.components.detection_panel.st.stop")
@patch("dashboard.components.detection_panel.st.success")
@patch("dashboard.components.detection_panel.st.error")
@patch("dashboard.components.detection_panel.st.metric")
@patch("dashboard.components.detection_panel.st.caption")
def test_render_detection_panel_normal(mock_cap, mock_met, mock_err, mock_succ, mock_stop):
    render_detection_panel(None, None)
    mock_succ.assert_called_once_with("✅ No anomaly detected")
    mock_stop.assert_called_once()
    mock_err.assert_not_called()

@patch("dashboard.components.detection_panel.st.stop")
@patch("dashboard.components.detection_panel.st.success")
@patch("dashboard.components.detection_panel.st.error")
@patch("dashboard.components.detection_panel.st.metric")
@patch("dashboard.components.detection_panel.st.caption")
def test_render_detection_panel_fault(mock_cap, mock_met, mock_err, mock_succ, mock_stop):
    render_detection_panel("Antenna Failure", 0.87)
    mock_err.assert_called_once_with("🚨 Fault Detected: Antenna Failure")
    mock_met.assert_called_once_with(label="Model Confidence", value="87.0%")
    mock_cap.assert_called_once()
    mock_succ.assert_not_called()
    mock_stop.assert_not_called()
