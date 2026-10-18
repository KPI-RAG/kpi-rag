import streamlit as st
from src.schema import RetrievedTicket

def render_sources_panel(tickets: list[RetrievedTicket], low_confidence: bool = False) -> None:
    if not tickets:
        st.info("No similar historical incidents retrieved")
        return

    st.subheader("Similar Historical Incidents")
    if low_confidence:   # these incidents were still given to the LLM
        st.warning("Low similarity to past incidents — treat them as weak precedent")
    for i, ticket in enumerate(tickets[:5]):
        with st.expander(f"[{i+1}] {ticket.anomaly_type} (similarity: {ticket.similarity_score:.2f})"):
            st.write(ticket.content[:500])
            st.caption(f"Ticket ID: {ticket.ticket_id}")
