import logging
import chromadb
from sentence_transformers import SentenceTransformer
from src.schema import ClassifierOutput, RetrievedTicket

logger = logging.getLogger(__name__)

def build_query(payload: ClassifierOutput) -> str:
    sorted_shap = sorted(payload.shap_top3, key=lambda x: abs(x.shap_value), reverse=True)
    
    top3_channels = ", ".join(x.channel for x in sorted_shap)
    
    dirs = []
    for x in sorted_shap:
        direction = "above normal" if x.shap_value > 0 else "below normal"
        dirs.append(f"{x.channel}: {direction}")
    directions = ", ".join(dirs)
    
    # signal_statistics is now flat: {"RSRP_mean": -77.6, "DL_BLER_mean": 0.35, ...}
    proto_states = []
    for k, v in payload.signal_statistics.items():
        if k.endswith("_mean") and ("UL" in k or "DL" in k):
            channel = k[: -len("_mean")]
            proto_states.append(f"{channel}: {v:.2f}")
            
    protocol_state = ", ".join(proto_states) if proto_states else "N/A"
    
    query = (
        f"{payload.anomaly_type.value} anomaly detected.\n"
        f"   Primary affected KPIs: {top3_channels}.\n"
        f"   Signal direction: {directions}.\n"
        f"   Protocol state: {protocol_state}."
    )
    return query

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model_name: str,
    k: int = 5,
    threshold: float = 0.45
) -> tuple[list[RetrievedTicket], bool]:
    model = SentenceTransformer(model_name)
    query_emb = model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k
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
    query_str = build_query(payload)
    model_name = cfg["rag"]["embedding_model"]
    k = cfg["rag"]["top_k"]
    threshold = cfg["rag"]["cosine_threshold"]
    
    return retrieve(
        query=query_str,
        collection=collection,
        model_name=model_name,
        k=k,
        threshold=threshold
    )
