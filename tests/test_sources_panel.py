import pytest
from unittest.mock import patch, MagicMock
from dashboard.components.sources_panel import render_sources_panel
from src.schema import RetrievedTicket

@patch("dashboard.components.sources_panel.st.info")
@patch("dashboard.components.sources_panel.st.subheader")
@patch("dashboard.components.sources_panel.st.expander")
@patch("dashboard.components.sources_panel.st.write")
@patch("dashboard.components.sources_panel.st.caption")
def test_render_sources_panel_empty_or_low_conf(mock_cap, mock_write, mock_exp, mock_sub, mock_info):
    render_sources_panel([], low_confidence=False)
    mock_info.assert_called_with("No similar historical incidents retrieved")
    
    render_sources_panel([MagicMock()], low_confidence=True)
    mock_info.assert_called_with("No similar historical incidents retrieved")
    
    assert mock_sub.call_count == 0

@patch("dashboard.components.sources_panel.st.info")
@patch("dashboard.components.sources_panel.st.subheader")
@patch("dashboard.components.sources_panel.st.expander")
@patch("dashboard.components.sources_panel.st.write")
@patch("dashboard.components.sources_panel.st.caption")
def test_render_sources_panel_valid(mock_cap, mock_write, mock_exp, mock_sub, mock_info):
    t1 = RetrievedTicket(ticket_id="T1", content="abc"*200, anomaly_type="Antenna Failure", similarity_score=0.912)
    t2 = RetrievedTicket(ticket_id="T2", content="def", anomaly_type="Cell Outage", similarity_score=0.888)
    
    render_sources_panel([t1, t2], low_confidence=False)
    
    mock_info.assert_not_called()
    mock_sub.assert_called_once_with("Similar Historical Incidents")
    
    mock_exp.assert_any_call("[1] Antenna Failure (similarity: 0.91)")
    mock_exp.assert_any_call("[2] Cell Outage (similarity: 0.89)")
