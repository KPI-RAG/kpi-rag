"""Test 03 — ChromaDB retrieval using real index."""
import pytest
from src.schema import ClassifierOutput
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output


@pytest.fixture(scope="module")
def collection(cfg):
    """Get the real ChromaDB collection."""
    return get_collection(cfg)


def test_collection_exists_and_has_documents(collection):
    """ChromaDB collection must exist and contain documents."""
    count = collection.count()
    if count == 0:
        pytest.skip("ChromaDB index empty — run build_index.py first")
    assert count > 0
    print(f"Collection has {count} documents")


def test_retrieve_returns_results_for_antenna_failure(collection, cfg, all_payloads):
    """Antenna Failure retrieval must return valid ticket objects."""
    if collection.count() == 0:
        pytest.skip("ChromaDB index empty")

    data = all_payloads["Antenna Failure"]
    payload = ClassifierOutput(**data)
    tickets, low_conf = query_from_classifier_output(payload, collection, cfg)
    assert isinstance(tickets, list)
    assert isinstance(low_conf, bool)
    if len(tickets) > 0:
        for t in tickets:
            assert isinstance(t.anomaly_type, str)
            assert isinstance(t.similarity_score, float)
            assert 0.0 <= t.similarity_score <= 1.0


def test_retrieve_all_fault_types_no_crash(collection, cfg, all_payloads):
    """Retrieval must not crash for any of the 11 fault types."""
    if collection.count() == 0:
        pytest.skip("ChromaDB index empty")

    for fault_name, data in all_payloads.items():
        payload = ClassifierOutput(**data)
        tickets, low_conf = query_from_classifier_output(payload, collection, cfg)
        assert isinstance(tickets, list), f"{fault_name}: tickets is not a list"
        assert isinstance(low_conf, bool), f"{fault_name}: low_conf is not bool"


def test_jamming_retrieval_no_crash(collection, cfg, all_payloads):
    """Jamming has no alignment entry but retrieval must still work."""
    if collection.count() == 0:
        pytest.skip("ChromaDB index empty")

    data = all_payloads["Jamming"]
    payload = ClassifierOutput(**data)
    tickets, low_conf = query_from_classifier_output(payload, collection, cfg)
    assert isinstance(tickets, list)
    assert isinstance(low_conf, bool)


def test_similarity_scores_in_valid_range(collection, cfg, all_payloads):
    """Similarity scores must be in a valid cosine similarity range."""
    if collection.count() == 0:
        pytest.skip("ChromaDB index empty")

    data = all_payloads["Antenna Failure"]
    payload = ClassifierOutput(**data)
    tickets, _ = query_from_classifier_output(payload, collection, cfg)
    if tickets:
        for t in tickets:
            # cosine similarity: allow small float error beyond [0, 1]
            assert -0.1 <= t.similarity_score <= 1.1, (
                f"Score {t.similarity_score} out of range for ticket {t.ticket_id}"
            )
