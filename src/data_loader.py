import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def load_jsonl_files(raw_path: str) -> list[dict]:
    """Load all JSONL files found recursively under a directory.

    Walks *raw_path* with ``rglob("*.jsonl")`` and deserialises every
    non-empty line as a JSON object.  A warning is logged when the
    directory does not exist; an info message reports the total number of
    records loaded.

    Parameters
    ----------
    raw_path : str
        Path to the root directory that contains (or recursively contains)
        ``*.jsonl`` files.

    Returns
    -------
    list[dict]
        A flat list of all JSON objects parsed from every JSONL file found
        under *raw_path*.  Returns an empty list if the directory does not
        exist or no JSONL files are present.
    """
    records = []
    path = Path(raw_path)
    if not path.exists():
        logger.warning("Path %s does not exist", raw_path)
        return records
        
    for file_path in path.rglob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), raw_path)
    return records

def filter_anomalous(records: list[dict]) -> list[dict]:
    """Filter records to keep only those flagged as anomalous.

    Checks each record's ``anomalies.exists`` field and retains only the
    records where that value is exactly ``True``.  Logs an info message
    with the count of anomalous records versus the total.

    Parameters
    ----------
    records : list[dict]
        Raw records as returned by :func:`load_jsonl_files`.  Each record
        is expected to be a dictionary that may contain an ``"anomalies"``
        sub-dictionary with an ``"exists"`` boolean key.

    Returns
    -------
    list[dict]
        Subset of *records* for which ``record["anomalies"]["exists"]``
        is ``True``.  Returns an empty list if no anomalous records are
        found.
    """
    anomalous = []
    for r in records:
        if r.get("anomalies", {}).get("exists") is True:
            anomalous.append(r)
    logger.info("Found %d anomalous records out of %d total", len(anomalous), len(records))
    return anomalous

def extract_tickets(records: list[dict]) -> list[dict]:
    """Extract and normalise ticket information from anomalous records.

    Iterates over *records* and builds a flat ticket dictionary for each
    one by pulling relevant fields out of the nested record structure.
    Missing fields are replaced with empty strings or the record's
    positional index as a fallback ticket ID.

    Parameters
    ----------
    records : list[dict]
        Anomalous records, typically the output of :func:`filter_anomalous`.
        Each record may contain the keys ``"ticket_id"``, ``"description"``,
        ``"anomalies"`` (with sub-keys ``"type"`` and
        ``"troubleshooting_tickets"``), and ``"QnA"`` (with sub-key
        ``"anomalies"``).

    Returns
    -------
    list[dict]
        One dictionary per input record with the following string-valued
        keys:

        * ``"ticket_id"``    – unique identifier for the ticket.
        * ``"anomaly_type"`` – category/type of the anomaly.
        * ``"ticket_text"``  – troubleshooting ticket narrative.
        * ``"qna_trace"``    – Q&A trace associated with the anomaly.
        * ``"description"``  – free-text description of the record.
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
    """Filter tickets to include only those in the training split.
    
    Reads a numpy array of training indices from `idx_path` and retains
    tickets whose integer `ticket_id` is present in that array. If the
    index file does not exist, all tickets are returned.
    
    Parameters
    ----------
    tickets : list[dict]
        List of ticket dictionaries.
    idx_path : str
        Path to the numpy file (.npy) containing training indices.
        
    Returns
    -------
    list[dict]
        Subset of `tickets` that belong to the training split.
    """
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

