import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def load_jsonl_files(raw_path: str) -> list[dict]:
    """All records from every *.jsonl under raw_path, in sorted file order (order matters: see apply_train_split)."""
    records = []
    path = Path(raw_path)
    if not path.exists():
        logger.warning("Path %s does not exist", raw_path)
        return records
        
    for file_path in sorted(path.rglob("*.jsonl")):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), raw_path)
    return records

def filter_anomalous(records: list[dict]) -> list[dict]:
    """Records whose anomalies.exists is True."""
    anomalous = []
    for r in records:
        if r.get("anomalies", {}).get("exists") is True:
            anomalous.append(r)
    logger.info("Found %d anomalous records out of %d total", len(anomalous), len(records))
    return anomalous

def extract_tickets(records: list[dict]) -> list[dict]:
    """Flatten anomalous records into tickets.

    TelecomTS records carry no id, so ticket_id is the record's position in
    the list (as a string).
    """
    tickets = []
    for i, r in enumerate(records):
        ticket_id = r.get("ticket_id", str(i))
        anomaly_type = r.get("anomalies", {}).get("type", "Unknown")
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
    """Keep the tickets whose id is in the train split (.npy of window indices).

    ticket_id equals the global window index only because data/raw/anomalous/
    sorts before data/raw/normal/ and files are read in sorted order.
    verify_ticket_order() checks that against the ground-truth file.
    Raises FileNotFoundError rather than returning everything if the split
    file is missing.
    """
    path = Path(idx_path)
    if not path.exists():
        # Never fail open here: indexing everything would put held-out windows in the corpus.
        raise FileNotFoundError(f"Train split file not found: {idx_path}")

    train_indices = set(np.load(path).tolist())
    
    retained = []
    for t in tickets:
        try:
            tid = int(t["ticket_id"])
            if tid in train_indices:
                retained.append(t)
        except ValueError:
            pass   # non-numeric id can't be in the split
            
    logger.info("Retained %d out of %d tickets after train split", len(retained), len(tickets))
    return retained


def verify_ticket_order(tickets: list[dict], handoff_path: str) -> None:
    """Raise ValueError unless ticket k's label matches the ground truth for window k in the handoff file."""
    with open(handoff_path, "r", encoding="utf-8") as f:
        truth = {int(r["window_index"]): r["ground_truth_anomaly_type"] for r in json.load(f)}
    bad = [t["ticket_id"] for t in tickets
           if int(t["ticket_id"]) in truth and truth[int(t["ticket_id"])] != t["anomaly_type"]]
    if bad:
        raise ValueError(
            f"{len(bad)} tickets do not match the ground-truth label for their window index "
            f"(e.g. {bad[:5]}); raw file order differs from the split files"
        )
    logger.info("Ticket order verified against %s (%d windows)", handoff_path, len(truth))

