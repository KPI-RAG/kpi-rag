import pytest
import chromadb
from src.schema import ClassifierOutput, RetrievedTicket
from src.rag_query import build_query, retrieve, query_from_classifier_output

@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chromadb"))
    # Use cosine distance to ensure similarity = 1 - distance works predictably.
    col = client.get_or_create_collection(name="test_col", metadata={"hnsw:space": "cosine"})
    
    # We want a perfect match to avoid low_confidence
    docs = [
        "Antenna Failure anomaly detected.\n   Primary affected KPIs: RSRP, DL_BLER, DL_MCS.\n   Signal direction: RSRP: below normal, DL_BLER: above normal, DL_MCS: below normal.\n   Protocol state: DL_BLER: 0.35.",
        "Random irrelevant text 1 about 5G",
        "Random irrelevant text 2 about 4G",
        "Random irrelevant text 3 about 3G",
        "Random irrelevant text 4 about 2G",
    ]
    metadatas = [
        {"ticket_id": "T1", "anomaly_type": "Antenna Failure"},
        {"ticket_id": "T2", "anomaly_type": "Other"},
        {"ticket_id": "T3", "anomaly_type": "Other"},
        {"ticket_id": "T4", "anomaly_type": "Other"},
        {"ticket_id": "T5", "anomaly_type": "Other"},
    ]
    ids = [f"T{i}" for i in range(1, 6)]
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(docs).tolist()
    
    col.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embs)
    return col

@pytest.fixture
def sample_payload():
    return ClassifierOutput(**{
        "anomaly_type": "Antenna Failure",
        "confidence": 0.87,
        "shap_top3": [
            {"channel": "RSRP",    "shap_value": -0.42, "feature_vs_normal": "below_normal_mean"},
            {"channel": "DL_BLER", "shap_value":  0.28, "feature_vs_normal": "above_normal_mean"},
            {"channel": "DL_MCS",  "shap_value": -0.19, "feature_vs_normal": "below_normal_mean"}
        ],
        "signal_statistics": {
            "RSRP_mean": -105, "RSRP_std": 3.2,  "RSRP_min": -112, "RSRP_max": -98,
            "DL_BLER_mean": 0.35, "DL_BLER_std": 0.08, "DL_BLER_min": 0.21, "DL_BLER_max": 0.51,
        }
    })

def test_build_query(sample_payload):
    query = build_query(sample_payload)
    assert isinstance(query, str)
    assert "Antenna Failure anomaly detected." in query
    assert "RSRP, DL_BLER, DL_MCS" in query
    assert "RSRP: below normal" in query
    assert "DL_BLER: above normal" in query
    assert "DL_MCS: below normal" in query
    assert "DL_BLER: 0.35" in query
    assert "RSRP" not in query.split("Protocol state:")[1] # RSRP doesn't have UL/DL

def test_retrieve(collection, sample_payload):
    query = build_query(sample_payload)
    tickets, low_conf = retrieve(query, collection, "all-MiniLM-L6-v2", k=2, threshold=0.45)
    
    assert isinstance(tickets, list)
    assert isinstance(low_conf, bool)
    assert len(tickets) > 0
    assert isinstance(tickets[0], RetrievedTicket)
    assert 0.0 <= tickets[0].similarity_score <= 1.0001
    
    # We seeded the DB with the exact query text
    assert low_conf is False

def test_retrieve_empty_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "empty_db"))
    col = client.get_or_create_collection(name="empty")
    tickets, low_conf = retrieve("test query", col, "all-MiniLM-L6-v2")
    assert len(tickets) == 0
    assert low_conf is True

def test_query_from_classifier_output(collection, sample_payload):
    cfg = {
        "rag": {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 3,
            "cosine_threshold": 0.45
        }
    }
    tickets, low_conf = query_from_classifier_output(sample_payload, collection, cfg)
    assert isinstance(tickets, list)
    assert isinstance(low_conf, bool)
    assert len(tickets) > 0
    assert low_conf is False


# ─── P2-B Negative retrieval test ────────────────────────────────────────────

def test_retrieval_low_similarity_for_unrelated_query(collection):
    """An off-topic query must return low cosine similarity against the telecom DB.

    The test collection contains only 5G/4G/3G/2G network fault descriptions.
    A food-domain query should land far from all indexed embeddings, so every
    returned similarity score must be well below 0.5.
    """
    from src.rag_query import retrieve
    tickets, low_conf = retrieve(
        "recipes for chocolate cake baking temperature and butter ratio",
        collection,
        model_name="all-MiniLM-L6-v2",
        k=3,
        threshold=0.45,
    )
    # The retriever should flag this as low-confidence
    assert low_conf is True, (
        "An off-topic query should trigger low_confidence=True"
    )
    # Every individual score must be below 0.5 — no spurious high-similarity hits
    for t in tickets:
        assert t.similarity_score < 0.5, (
            f"Unrelated query returned suspiciously high similarity "
            f"{t.similarity_score:.3f} for ticket '{t.ticket_id}'. "
            "Retriever may not be discriminating correctly."
        )
