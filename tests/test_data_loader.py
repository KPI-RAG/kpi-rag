import pytest
import numpy as np
import json
from src.data_loader import load_jsonl_files, filter_anomalous, extract_tickets, apply_train_split

@pytest.fixture
def fake_data():
    return [
        {
            "ticket_id": "0",
            "anomalies": {"exists": False},
            "labels": "Normal",
            "description": "normal desc",
            "QnA": {"anomalies": ""}
        },
        {
            "ticket_id": "1",
            "anomalies": {"exists": True, "troubleshooting_tickets": "Fix X"},
            "labels": "Antenna Failure",
            "description": "desc 1",
            "QnA": {"anomalies": "trace 1"}
        },
        {
            "ticket_id": "2",
            "anomalies": {"exists": True, "troubleshooting_tickets": "Fix Y"},
            "labels": "Jamming",
            "description": "desc 2",
            "QnA": {"anomalies": "trace 2"}
        }
    ]

def test_load_jsonl_files(tmp_path, fake_data):
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    file_path = raw_path / "data.jsonl"
    with open(file_path, "w") as f:
        for r in fake_data:
            f.write(json.dumps(r) + "\n")
            
    records = load_jsonl_files(str(raw_path))
    assert len(records) == 3
    assert isinstance(records, list)

def test_filter_anomalous(fake_data):
    anomalous = filter_anomalous(fake_data)
    assert len(anomalous) == 2
    for r in anomalous:
        assert r["anomalies"]["exists"] is True
    # No normal sample in output
    assert not any(r["anomalies"]["exists"] is False for r in anomalous)

def test_extract_tickets(fake_data):
    anomalous = filter_anomalous(fake_data)
    tickets = extract_tickets(anomalous)
    assert len(tickets) == 2
    for t in tickets:
        assert "ticket_id" in t
        assert "anomaly_type" in t
        assert "ticket_text" in t
        assert "qna_trace" in t
        assert "description" in t
        
    assert tickets[0]["anomaly_type"] == "Antenna Failure"
    assert tickets[0]["ticket_text"] == "Fix X"

def test_apply_train_split(tmp_path, fake_data):
    anomalous = filter_anomalous(fake_data)
    tickets = extract_tickets(anomalous)
    
    idx_path = tmp_path / "train_idx.npy"
    np.save(idx_path, np.array([1]))
    
    retained = apply_train_split(tickets, str(idx_path))
    assert len(retained) == 1
    assert retained[0]["ticket_id"] == "1"
