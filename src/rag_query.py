import logging
import chromadb
from src.kg_indexer import get_model as _get_model
from src.schema import ClassifierOutput, RetrievedTicket

logger = logging.getLogger(__name__)

def build_query(payload: ClassifierOutput) -> str:
    """Build a text query for the RAG system from a ClassifierOutput payload.
    
    Parameters
    ----------
    payload : ClassifierOutput
        The anomaly detection payload.
        
    Returns
    -------
    str
        A formatted string describing the anomaly and KPIs.
    """
    sorted_shap = sorted(payload.shap_top3, key=lambda x: abs(x.shap_value), reverse=True)

    # Several top SHAP features can belong to one channel (e.g. RSRP_max/mean/min);
    # keep the strongest entry per channel so the query doesn't repeat it.
    seen = set()
    unique_shap = []
    for x in sorted_shap:
        if x.channel not in seen:
            seen.add(x.channel)
            unique_shap.append(x)

    top3_channels = ", ".join(x.channel for x in unique_shap)

    dirs = []
    for x in unique_shap:
        direction = "above normal" if "above" in x.feature_vs_normal else "below normal"
        dirs.append(f"{x.channel}: {direction}")
    directions = ", ".join(dirs)
    
    # signal_statistics is now flat: {"RSRP_mean": -77.6, "DL_BLER_mean": 0.35, ...}
    proto_states = []
    for k, v in payload.signal_statistics.items():
        if k.endswith("_mean") and ("UL" in k or "DL" in k):
            channel = k[: -len("_mean")]
            proto_states.append(f"{channel}: {v:.2f}")
            
    protocol_state = ", ".join(proto_states) if proto_states else "N/A"
    
    # No fault label in the query: the class is applied as a metadata filter in
    # query_from_classifier_output(), so similarity ranks incidents by KPI evidence only.
    query = (
        f"Primary affected KPIs: {top3_channels}.\n"
        f"Signal direction: {directions}.\n"
        f"Protocol state: {protocol_state}."
    )
    return query

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model_name: str,
    k: int = 5,
    threshold: float = 0.45,
    where: dict | None = None,
) -> tuple[list[RetrievedTicket], bool]:
    """Retrieve similar tickets from ChromaDB using a text query.
    
    Parameters
    ----------
    query : str
        The query text to embed and search.
    collection : chromadb.Collection
        The ChromaDB collection to search in.
    model_name : str
        The SentenceTransformer model name.
    k : int, optional
        Number of tickets to retrieve (default is 5).
    threshold : float, optional
        Minimum cosine similarity threshold to be considered high confidence (default is 0.45).
    where : dict, optional
        ChromaDB metadata filter, e.g. {"anomaly_type": "Antenna Failure"}.

    Returns
    -------
    tuple[list[RetrievedTicket], bool]
        A tuple containing the list of retrieved tickets and a boolean indicating
        if the retrieval was low confidence (highest score < threshold).
    """
    query_emb = _get_model(model_name).encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where=where,
    )
    
    tickets = []
    if not results["ids"] or not results["ids"][0]:
        logger.info("Query '%s' (len %d): no tickets retrieved. low_confidence=True", query[:20], len(query))
        return tickets, True
        
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else [0.0]*len(ids)
    
    top_score = -1.0
    
    for i in range(len(ids)):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        tid = meta.get("ticket_id", str(i)) if meta else str(i)
        content = documents[i] if documents and i < len(documents) else ""
        anomaly = meta.get("anomaly_type", "unknown") if meta else "unknown"
        
        similarity = float(1.0 - distances[i])
        
        if similarity > top_score:
            top_score = similarity
            
        ticket = RetrievedTicket(
            ticket_id=str(tid),
            content=str(content),
            anomaly_type=str(anomaly),
            similarity_score=similarity
        )
        tickets.append(ticket)
        
    low_confidence = False
    if len(tickets) == 0 or top_score < threshold:
        low_confidence = True
        
    logger.info("Query '%s' (len %d): top_score=%.4f, low_confidence=%s", 
                query[:20], len(query), top_score, low_confidence)
                
    return tickets, low_confidence

def query_from_classifier_output(
    payload: ClassifierOutput,
    collection: chromadb.Collection,
    cfg: dict
) -> tuple[list[RetrievedTicket], bool]:
    """Past incidents of the predicted fault type, ranked by KPI-evidence similarity.

    Text retrieval cannot tell the fault classes apart (measured: P@1 vs the true
    label 0.45 without the label, below the classifier's 0.75), so the class comes
    from the classifier as a filter and retrieval only ranks evidence within it.

    Parameters
    ----------
    payload : ClassifierOutput
        The anomaly detection payload.
    collection : chromadb.Collection
        The ChromaDB collection to search in.
    cfg : dict
        Application configuration containing RAG settings.
        
    Returns
    -------
    tuple[list[RetrievedTicket], bool]
        A tuple containing the list of retrieved tickets and a boolean indicating
        if the retrieval was low confidence.
    """
    query_str = build_query(payload)
    model_name = cfg["rag"]["embedding_model"]
    k = cfg["rag"]["top_k"]
    threshold = cfg["rag"]["cosine_threshold"]
    
    return retrieve(
        query=query_str,
        collection=collection,
        model_name=model_name,
        k=k,
        threshold=threshold,
        where={"anomaly_type": payload.anomaly_type.value},
    )


def retrieve_similar_tickets(
    query: str,
    cfg: dict,
    n_results: int = 5
) -> list[dict]:
    """Retrieve similar tickets for a free-text query and return a list of dictionaries.

    Parameters
    ----------
    query : str
        Free-text query to search.
    cfg : dict
        Application configuration containing RAG settings.
    n_results : int, optional
        Number of tickets to retrieve (default is 5).

    Returns
    -------
    list[dict]
        List of dicts with keys 'ticket_id', 'fault_type', 'similarity', 'content'.
    """
    from src.kg_indexer import get_collection
    collection = get_collection(cfg)
    model_name = cfg["rag"]["embedding_model"]
    threshold = cfg["rag"]["cosine_threshold"]
    tickets, _ = retrieve(
        query=query,
        collection=collection,
        model_name=model_name,
        k=n_results,
        threshold=threshold,
    )
    return [
        {
            "ticket_id": t.ticket_id,
            "fault_type": t.anomaly_type,
            "similarity": t.similarity_score,
            "content": t.content,
        }
        for t in tickets
    ]
