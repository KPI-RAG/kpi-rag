"""Retrieval design experiment: which document + query design retrieves
incidents of the TRUE fault type for held-out windows? In memory, no index changes, no LLM.

Evaluated on the test windows Phase 2 detected that are true faults (266). Needs data/raw.
    uv run python scripts/experiment_retrieval_design.py
"""
import json
import logging

import numpy as np

from src.config_loader import load_config
from src.data_loader import load_jsonl_files, filter_anomalous, extract_tickets, apply_train_split
from src.kg_indexer import get_model
from src.rag_query import build_query
from src.schema import ClassifierOutput
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def precision_at(query_emb: np.ndarray, doc_emb: np.ndarray, doc_labels: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    top = np.argsort(-(query_emb @ doc_emb.T), axis=1)[:, :5]
    return float((doc_labels[top[:, 0]] == truth).mean()), float((doc_labels[top] == truth[:, None]).mean())


def main() -> None:
    setup_logging(__name__)
    cfg = load_config()
    data = cfg["data"]
    tickets = extract_tickets(filter_anomalous(load_jsonl_files(data["raw_path"])))
    train = apply_train_split(tickets, f"{data['indices_path']}/{data['train_idx_file']}")
    labels = np.array([t["anomaly_type"] for t in train])

    with open(f"{data['processed_path']}/{data['layer2_output_file']}", encoding="utf-8") as f:
        layer2 = json.load(f)
    with open(f"{data['processed_path']}/layer2_rag_handoff_sessionsplit.json", encoding="utf-8") as f:
        truth_by_window = {r["window_index"]: r["ground_truth_anomaly_type"] for r in json.load(f)}
    evalset = [w for w in layer2 if w["anomaly_type"] != "Normal" and w["window_index"] in truth_by_window]
    payloads = [ClassifierOutput(**w) for w in evalset]
    truth = np.array([truth_by_window[w["window_index"]] for w in evalset])

    model = get_model(cfg["rag"]["embedding_model"])

    def emb(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=64)

    def paragraph_mean(texts):
        out = []
        for t in texts:
            v = emb([p for p in t.split("\n\n") if p.strip()] or [t]).mean(0)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    label_query = [f"{p.anomaly_type.value} anomaly detected.\n{build_query(p)}" for p in payloads]   # previous design
    evidence_query = [build_query(p) for p in payloads]                                                # label-free
    template_docs = emb([t["ticket_text"] + "\n" + t["qna_trace"] for t in train])                    # previous corpus
    description_docs = emb([t["description"] for t in train])

    rows = {
        "previous: template docs, label in query": precision_at(emb(label_query), template_docs, labels, truth),
        "template docs, label-free query": precision_at(emb(evidence_query), template_docs, labels, truth),
        "description docs, label-free query (deployed ranking)": precision_at(emb(evidence_query), description_docs, labels, truth),
        "description docs averaged over paragraphs": precision_at(emb(evidence_query), paragraph_mean([t["description"] for t in train]), labels, truth),
    }
    print(f"windows: {len(evalset)} | classifier exact accuracy on them: "
          f"{np.mean([p.anomaly_type.value == t for p, t in zip(payloads, truth)]):.3f} | "
          f"majority-class share: {max(np.mean(truth == c) for c in set(truth)):.3f}")
    print(f"distinct ticket_text templates: {len({t['ticket_text'] for t in train})}")
    for name, (p1, p5) in rows.items():
        print(f"  {name:55s} P@1 vs true class = {p1:.3f}   P@5 = {p5:.3f}")


if __name__ == "__main__":
    main()
