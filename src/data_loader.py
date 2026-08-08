import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def load_jsonl_files(raw_path: str) -> list[dict]:
    records = []
    path = Path(raw_path)
    if not path.exists():
        logger.warning("Path %s does not exist", raw_path)
        return records
        
    for file_path in path.glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), raw_path)
    return records

def filter_anomalous(records: list[dict]) -> list[dict]:
    anomalous = []
    for r in records:
        if r.get("anomalies", {}).get("exists") is True:
            anomalous.append(r)
    logger.info("Found %d anomalous records out of %d total", len(anomalous), len(records))
    return anomalous

def extract_tickets(records: list[dict]) -> list[dict]:
    tickets = []
    for i, r in enumerate(records):
        ticket_id = r.get("ticket_id", str(i))
        anomaly_type = r.get("labels", "Unknown")
        ticket_text = r.get("anomalies", {}).get("troubleshooting_tickets", "")
        qna_trace = r.get("QnA", {}).get("anomalies", "")
        description = r.get("description", "")
        
        tickets.append({
            "ticket_id": str(ticket_id),
            "anomaly_type": str(anomaly_type),
            "ticket_text": str(ticket_text),
            "qna_trace": str(qna_trace),
            "description": str(description)
        })
    return tickets

def apply_train_split(tickets: list[dict], idx_path: str) -> list[dict]:
    path = Path(idx_path)
    if not path.exists():
        logger.warning("Train idx file %s not found, returning all tickets", idx_path)
        return tickets
        
    train_indices = set(np.load(path).tolist())
    
    retained = []
    for t in tickets:
        try:
            tid = int(t["ticket_id"])
            if tid in train_indices:
                retained.append(t)
        except ValueError:
            # If ticket_id cannot be cast to int, we include it safely or exclude it?
            # Usually index arrays are integers. If it fails, assume it's not in the index array.
            pass
            
    logger.info("Retained %d out of %d tickets after train split", len(retained), len(tickets))
    return retained
