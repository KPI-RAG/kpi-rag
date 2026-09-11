import logging
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def get_collection(cfg: dict) -> chromadb.Collection:
    """Return (or create) the ChromaDB collection specified in the config.

    Connects to a persistent ChromaDB instance at the path defined in ``cfg``
    and either retrieves an existing collection or creates a new one with
    cosine-distance HNSW indexing.

    Parameters
    ----------
    cfg : dict
        Application configuration dictionary.  Must contain the nested keys
        ``cfg["rag"]["chroma_db_path"]`` (filesystem path to the ChromaDB
        directory) and ``cfg["rag"]["collection_name"]`` (name of the target
        collection).

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection object, ready for querying or ingestion.
    """
    path = cfg["rag"]["chroma_db_path"]
    name = cfg["rag"]["collection_name"]
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    logger.info("Loaded collection %s with %d documents", name, collection.count())
    return collection

def embed_tickets(tickets: list[dict], model_name: str) -> tuple[list[str], list[list[float]], list[dict], list[str]]:
    """Encode a list of support tickets into dense vector embeddings.

    Concatenates each ticket's ``ticket_text`` and ``qna_trace`` fields into a
    single document string, then encodes all documents in one batch using a
    ``SentenceTransformer`` model.

    Parameters
    ----------
    tickets : list[dict]
        List of ticket dictionaries.  Each dict must contain the keys
        ``"ticket_id"``, ``"ticket_text"``, ``"qna_trace"``, and
        ``"anomaly_type"``.
    model_name : str
        Name or path of the ``SentenceTransformer`` model used to compute
        embeddings (e.g. ``"all-MiniLM-L6-v2"``).

    Returns
    -------
    documents : list[str]
        Concatenated text strings, one per ticket.
    embeddings : list[list[float]]
        Dense embedding vectors corresponding to each document.
    metadatas : list[dict]
        Metadata dicts containing ``"ticket_id"`` and ``"anomaly_type"`` for
        each ticket.
    ids : list[str]
        Ticket ID strings used as unique identifiers in ChromaDB.
    """
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
    """Index a list of tickets into a ChromaDB collection.
    
    Filters out tickets that are already present in the collection by ID.
    
    Parameters
    ----------
    tickets : list[dict]
        List of ticket dictionaries to index.
    collection : chromadb.Collection
        The ChromaDB collection to index into.
    model_name : str
        The SentenceTransformer model name to use for embedding.
        
    Returns
    -------
    int
        The number of new tickets successfully indexed.
    """
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
    """Clear all documents from a ChromaDB collection.
    
    Parameters
    ----------
    collection : chromadb.Collection
        The collection to clear.
    """
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
