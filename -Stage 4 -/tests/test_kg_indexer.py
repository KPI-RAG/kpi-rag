import pytest
from src.kg_indexer import get_collection, embed_tickets, index_tickets, clear_collection

@pytest.fixture
def fake_cfg(tmp_path):
    return {
        "rag": {
            "chroma_db_path": str(tmp_path / "chroma_db"),
            "collection_name": "test_tickets",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    }

@pytest.fixture
def fake_tickets():
    return [
        {
            "ticket_id": "1",
            "anomaly_type": "Antenna Failure",
            "ticket_text": "Antenna 1 is down.",
            "qna_trace": "Because of wind.",
            "description": "test"
        },
        {
            "ticket_id": "2",
            "anomaly_type": "Jamming",
            "ticket_text": "Signal blocked.",
            "qna_trace": "Interference detected.",
            "description": "test"
        },
        {
            "ticket_id": "3",
            "anomaly_type": "Buffer Overflow (Gradual Buildup)",
            "ticket_text": "Buffer full.",
            "qna_trace": "Packets dropping.",
            "description": "test"
        }
    ]

def test_get_collection(fake_cfg):
    collection = get_collection(fake_cfg)
    assert collection.name == "test_tickets"

def test_embed_tickets(fake_tickets):
    model_name = "all-MiniLM-L6-v2"
    docs, embs, metas, ids = embed_tickets(fake_tickets, model_name)
    
    assert len(docs) == 3
    assert len(embs) == 3
    assert len(metas) == 3
    assert len(ids) == 3
    
    assert isinstance(embs[0], list)
    assert isinstance(embs[0][0], float)
    assert len(embs[0]) > 0
    assert docs[0] == "Antenna 1 is down.\nBecause of wind."
    assert metas[0]["ticket_id"] == "1"

def test_index_tickets(fake_cfg, fake_tickets):
    collection = get_collection(fake_cfg)
    model_name = fake_cfg["rag"]["embedding_model"]
    
    count1 = index_tickets(fake_tickets, collection, model_name)
    assert count1 == 3
    assert collection.count() == 3
    
    count2 = index_tickets(fake_tickets, collection, model_name)
    assert count2 == 0
    assert collection.count() == 3
    
def test_clear_collection(fake_cfg, fake_tickets):
    collection = get_collection(fake_cfg)
    model_name = fake_cfg["rag"]["embedding_model"]
    index_tickets(fake_tickets, collection, model_name)
    
    assert collection.count() == 3
    clear_collection(collection)
    assert collection.count() == 0
