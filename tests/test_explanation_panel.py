import pytest
from unittest.mock import patch
from dashboard.components.explanation_panel import render_explanation_panel
from src.schema import LLMExplanation

@patch("dashboard.components.explanation_panel.st.warning")
@patch("dashboard.components.explanation_panel.st.info")
@patch("dashboard.components.explanation_panel.st.subheader")
@patch("dashboard.components.explanation_panel.st.write")
def test_render_explanation_panel_empty(mock_write, mock_sub, mock_info, mock_warn):
    render_explanation_panel(None, low_confidence=True)
    mock_warn.assert_called_once()
    mock_info.assert_called_once_with("No explanation generated")
    mock_sub.assert_not_called()
    mock_write.assert_not_called()

@patch("dashboard.components.explanation_panel.st.warning")
@patch("dashboard.components.explanation_panel.st.info")
@patch("dashboard.components.explanation_panel.st.subheader")
@patch("dashboard.components.explanation_panel.st.write")
@patch("dashboard.components.explanation_panel.st.success")
@patch("dashboard.components.explanation_panel.st.error")
@patch("dashboard.components.explanation_panel.st.caption")
def test_render_explanation_panel_valid(mock_cap, mock_err, mock_succ, mock_write, mock_sub, mock_info, mock_warn):
    expl = LLMExplanation(
        root_cause="cause",
        gpp_reference="TS 38.104",
        oran_component="oran",
        recommended_action="action",
        reasoning_trace="trace",
        reference_valid=True,
        template_generated=True
    )
    
    render_explanation_panel(expl, low_confidence=False)
    
    mock_warn.assert_not_called()
    mock_info.assert_not_called()
    
    assert mock_sub.call_count == 5
    assert mock_write.call_count == 4
    
    mock_succ.assert_called_once_with("✅ TS 38.104")
    mock_err.assert_not_called()
    mock_cap.assert_called_once_with("📋 Generated from alignment table template (LLM parse failed)")

@patch("dashboard.components.explanation_panel.st.warning")
@patch("dashboard.components.explanation_panel.st.info")
@patch("dashboard.components.explanation_panel.st.subheader")
@patch("dashboard.components.explanation_panel.st.write")
@patch("dashboard.components.explanation_panel.st.success")
@patch("dashboard.components.explanation_panel.st.error")
@patch("dashboard.components.explanation_panel.st.caption")
def test_render_explanation_panel_invalid_ref(mock_cap, mock_err, mock_succ, mock_write, mock_sub, mock_info, mock_warn):
    expl = LLMExplanation(
        root_cause="cause",
        gpp_reference="TS 99.999",
        oran_component="oran",
        recommended_action="action",
        reasoning_trace="trace",
        reference_valid=False,
        template_generated=False
    )
    
    render_explanation_panel(expl, low_confidence=True)
    
    mock_warn.assert_called_once()
    mock_info.assert_not_called()
    mock_err.assert_called_once_with("❌ TS 99.999 (unvalidated)")
    mock_succ.assert_not_called()
    mock_cap.assert_not_called()
