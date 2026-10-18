from src.config_loader import load_config
from src.schema import ClassifierOutput, LLMExplanation
from src.utils import setup_logging
from src.kg_indexer import get_collection
from src.rag_query import query_from_classifier_output
from src.llm_explainer import load_alignment_table, explain, alignment_path
import argparse, json, logging, sys
from pydantic import ValidationError

logger = logging.getLogger(__name__)

def run(payload: ClassifierOutput, cfg: dict) -> LLMExplanation:
    collection = get_collection(cfg)
    tickets, low_conf = query_from_classifier_output(payload, collection, cfg)
    
    logger.info("Anomaly_type: %s, retrieved: %d tickets, low_conf: %s", 
                payload.anomaly_type.value, len(tickets), low_conf)
                
    if low_conf:
        logger.warning("Low retrieval confidence for %s", payload.anomaly_type.value)
        
    alignment = load_alignment_table(alignment_path(cfg))
    result = explain(payload, tickets, cfg, alignment)
    
    return result

def main() -> None:
    from dotenv import load_dotenv
    import os
    if os.path.exists(".env"):
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run complete pipeline")
    parser.add_argument("--input", type=str, required=True, help="path to ClassifierOutput JSON file")
    parser.add_argument("--output", type=str, required=True, help="path to write LLMExplanation JSON")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path to config.yaml")
    
    args = parser.parse_args()
    
    setup_logging(__name__)
    
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
        
    try:
        with open(args.input, encoding="utf-8") as f:
            payload = ClassifierOutput(**json.load(f))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Invalid ClassifierOutput JSON in %s: %s", args.input, e)
        sys.exit(1)
        
    result = run(payload, cfg)
    
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
    except Exception as e:
        logger.error("Error writing output: %s", e)
        sys.exit(1)
        
    logger.info("Explanation written to %s", args.output)

if __name__ == "__main__":
    main()
