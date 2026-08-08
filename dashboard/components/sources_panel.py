import streamlit as st
from src.schema import RetrievedTicket

def render_sources_panel(tickets: list[RetrievedTicket], low_confidence: bool = False) -> None:
    if low_confidence or not tickets:
        st.info("No similar historical incidents retrieved")
        return
        
    st.subheader("Similar Historical Incidents")
    for i, ticket in enumerate(tickets[:5]):
        with st.expander(f"[{i+1}] {ticket.anomaly_type} (similarity: {ticket.similarity_score:.2f})"):
            st.write(ticket.content[:500])
            st.caption(f"Ticket ID: {ticket.ticket_id}")
