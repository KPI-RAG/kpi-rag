import logging
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def get_collection(cfg: dict) -> chromadb.Collection:
    path = cfg["rag"]["chroma_db_path"]
    name = cfg["rag"]["collection_name"]
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    logger.info("Loaded collection %s with %d documents", name, collection.count())
    return collection

def embed_tickets(tickets: list[dict], model_name: str) -> tuple[list[str], list[list[float]], list[dict], list[str]]:
    logger.info("Embedding %d tickets using model %s", len(tickets), model_name)
    model = SentenceTransformer(model_name)
    
    documents = []
    metadatas = []
    ids = []
    
    for ticket in tickets:
        doc = f"{ticket['ticket_text']}\n{ticket['qna_trace']}"
        documents.append(doc)
        metadatas.append({
            "ticket_id": ticket["ticket_id"],
            "anomaly_type": ticket["anomaly_type"]
        })
        ids.append(ticket["ticket_id"])
        
    embeddings_np = model.encode(documents)
    embeddings = embeddings_np.tolist()
    
    return documents, embeddings, metadatas, ids

def index_tickets(tickets: list[dict], collection: chromadb.Collection, model_name: str) -> int:
    if not tickets:
        return 0
        
    documents, embeddings, metadatas, ids = embed_tickets(tickets, model_name)
    
    # Check for existing IDs to skip duplicates
    existing_result = collection.get(ids=ids)
    existing_ids = set(existing_result["ids"])
    
    new_docs, new_embs, new_metas, new_ids = [], [], [], []
    for doc, emb, meta, tid in zip(documents, embeddings, metadatas, ids):
        if tid not in existing_ids:
            new_docs.append(doc)
            new_embs.append(emb)
            new_metas.append(meta)
            new_ids.append(tid)
            
    if new_ids:
        collection.add(
            documents=new_docs,
            embeddings=new_embs,
            metadatas=new_metas,
            ids=new_ids
        )
        
    logger.info("Indexed %d new tickets out of %d provided", len(new_ids), len(tickets))
    return len(new_ids)

def clear_collection(collection: chromadb.Collection) -> None:
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
