import argparse
import logging
from src.config_loader import load_config
from src.data_loader import (
    load_jsonl_files,
    filter_anomalous,
    extract_tickets,
    apply_train_split
)
from src.kg_indexer import get_collection, index_tickets
from src.utils import setup_logging

logger = logging.getLogger(__name__)

def build_index(cfg: dict) -> int:
    raw_path = cfg["data"]["raw_path"]
    indices_path = cfg["data"]["indices_path"]
    embedding_model = cfg["rag"]["embedding_model"]
    
    records = load_jsonl_files(raw_path)
    logger.info("Loaded %d raw records", len(records))
    
    anomalous = filter_anomalous(records)
    logger.info("Filtered to %d anomalous records", len(anomalous))
    
    tickets = extract_tickets(anomalous)
    logger.info("Extracted %d tickets", len(tickets))
    
    train_tickets = apply_train_split(tickets, indices_path)
    logger.info("Applied train split, %d tickets remaining", len(train_tickets))
    
    collection = get_collection(cfg)
    logger.info("Got collection %s", collection.name if hasattr(collection, 'name') else 'unknown')
    
    indexed_count = index_tickets(train_tickets, collection, embedding_model)
    logger.info("Indexed %d documents", indexed_count)
    
    return indexed_count

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from TelecomTS data")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    setup_logging(__name__)
    
    cfg = load_config(args.config)
    indexed_count = build_index(cfg)
    logger.info("Indexed %d tickets into ChromaDB", indexed_count)

if __name__ == "__main__":
    main()
