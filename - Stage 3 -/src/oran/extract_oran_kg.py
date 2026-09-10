import os
import json
import sqlite3
import ijson
from datetime import datetime
from decimal import Decimal


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = r"data\oran\ORAN_Spec_Knowledge_graph.json"

OUTPUT_DIR = r"data\oran\extracted"

DB_FILE = os.path.join(OUTPUT_DIR, "oran_kg.sqlite")

NODES_FILE = os.path.join(OUTPUT_DIR, "oran_nodes.jsonl")
RELATIONSHIPS_FILE = os.path.join(OUTPUT_DIR, "oran_relationships.jsonl")
RELEVANT_FILE = os.path.join(OUTPUT_DIR, "oran_relevant_nodes.jsonl")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "extraction_summary.json")


# ============================================================
# Fields that should NOT be copied
# ============================================================

# The embeddings are extremely large and are not needed for
# the first KPI-RAG extraction.
SKIP_PROPERTIES = {
    "embedding",
}


# ============================================================
# Keywords relevant to KPI-RAG / O-RAN
# ============================================================

RELEVANT_KEYWORDS = [
    # O-RAN
    "o-ran",
    "oran",
    "open ran",
    "o-ru",
    "oru",
    "o-du",
    "odu",
    "o-cu",
    "ocu",

    # RIC
    "near-rt ric",
    "near rt ric",
    "non-rt ric",
    "non rt ric",
    "ric",
    "xapp",
    "rapp",

    # O-RAN interfaces
    "e2",
    "a1",
    "o1",
    "o2",
    "open fronthaul",
    "fronthaul",

    # Network components
    "network component",
    "networkcomponent",
    "network function",
    "networkfunction",
    "logical node",
    "logicalnode",
    "component",
    "device",
    "node",

    # Telecom / network concepts
    "kpi",
    "metric",
    "indicator",
    "performance",
    "radio",
    "ran",
    "5g",
    "4g",
    "telecom",
    "telecommunications",

    # Standards
    "3gpp",
    "specification",
    "standard",
    "architecture",
    "interface",
    "protocol",
]


# ============================================================
# Utility functions
# ============================================================

def clean_value(value):
    """
    Convert values into JSON-safe lightweight representations.

    Decimal values are converted to float.
    Embeddings are removed separately.
    """
    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {
            str(k): clean_value(v)
            for k, v in value.items()
            if str(k) not in SKIP_PROPERTIES
        }

    if isinstance(value, list):
        return [clean_value(v) for v in value]

    return value


def clean_properties(properties):
    """
    Remove embeddings and convert Decimal values.
    """
    if not isinstance(properties, dict):
        return {}

    cleaned = {}

    for key, value in properties.items():

        if key in SKIP_PROPERTIES:
            continue

        cleaned[key] = clean_value(value)

    return cleaned


def get_text_from_node(node):
    """
    Build searchable text from the node labels and properties.
    """

    if not isinstance(node, dict):
        return ""

    labels = node.get("labels", [])
    properties = node.get("properties", {})

    parts = []

    if labels:
        parts.extend(str(label) for label in labels)

    if isinstance(properties, dict):

        # These are the most useful properties for semantic retrieval.
        preferred_fields = [
            "name",
            "title",
            "text",
            "description",
            "definition",
            "content",
            "fileName",
            "type",
            "category",
            "abbreviation",
            "acronym",
            "term",
            "technology",
            "component",
            "function",
            "system",
            "interface",
            "protocol",
            "standard",
            "specification",
        ]

        for field in preferred_fields:

            if field not in properties:
                continue

            value = properties[field]

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                parts.append(str(value))

    return " ".join(parts)


def is_relevant(node):
    """
    Determine whether a node is potentially useful for KPI-RAG.
    """

    labels = node.get("labels", [])
    properties = node.get("properties", {})

    searchable_parts = []

    searchable_parts.extend(str(x) for x in labels)

    if isinstance(properties, dict):

        for key, value in properties.items():

            if key in SKIP_PROPERTIES:
                continue

            searchable_parts.append(str(key))

            if isinstance(value, (str, int, float, bool)):
                searchable_parts.append(str(value))

    text = " ".join(searchable_parts).lower()

    return any(keyword in text for keyword in RELEVANT_KEYWORDS)


def extract_relationship_info(record):
    """
    Try to identify relationship metadata if it exists in the exported
    record.

    The current O-RAN dataset appears to primarily expose node_start and
    node_end. Therefore, if no explicit relationship type exists, we
    preserve the connection with relationship_type = None rather than
    inventing one.
    """

    possible_keys = [
        "relationship",
        "rel",
        "relationship_data",
        "edge",
        "relationship_start",
        "relationship_end",
    ]

    relationship = None

    for key in possible_keys:

        if key in record:
            relationship = record[key]
            break

    if isinstance(relationship, dict):

        relationship_type = (
            relationship.get("type")
            or relationship.get("relationship_type")
            or relationship.get("label")
            or relationship.get("name")
        )

        properties = relationship.get("properties", {})

        return relationship_type, clean_properties(properties)

    if isinstance(relationship, str):
        return relationship, {}

    return None, {}


# ============================================================
# Main extraction
# ============================================================

def main():

    start_time = datetime.now()

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_FILE}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("O-RAN Knowledge Graph Extractor")
    print("=" * 70)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    file_size_gb = os.path.getsize(INPUT_FILE) / (1024 ** 3)

    print(f"Input file size: {file_size_gb:.2f} GB")
    print()
    print("The file will be processed STREAMING.")
    print("The complete 3.38 GB JSON will NOT be loaded into RAM.")
    print("Embeddings will NOT be copied.")
    print()

    # --------------------------------------------------------
    # SQLite database
    # --------------------------------------------------------

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute("""
        PRAGMA journal_mode = WAL;
    """)

    cursor.execute("""
        PRAGMA synchronous = NORMAL;
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            element_id TEXT PRIMARY KEY,
            labels TEXT,
            properties TEXT,
            search_text TEXT,
            relevant INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_id TEXT,
            end_id TEXT,
            start_labels TEXT,
            end_labels TEXT,
            relationship_type TEXT,
            properties TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodes_relevant
        ON nodes(relevant)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relationship_start
        ON relationships(start_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relationship_end
        ON relationships(end_id)
    """)

    conn.commit()

    # --------------------------------------------------------
    # JSONL outputs
    # --------------------------------------------------------

    nodes_out = open(
        NODES_FILE,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    )

    relevant_out = open(
        RELEVANT_FILE,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    )

    relationships_out = open(
        RELATIONSHIPS_FILE,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    records_processed = 0
    nodes_seen = 0
    nodes_new = 0
    relevant_nodes = 0
    relationships = 0

    label_counts = {}

    # --------------------------------------------------------
    # Streaming parser
    # --------------------------------------------------------

    print("Starting extraction...")
    print("This may take some time because the source is 3.38 GB.")
    print()

    with open(INPUT_FILE, "rb") as f:

        items = ijson.items(f, "item")

        for record in items:

            records_processed += 1

            if not isinstance(record, dict):
                continue

            start_node = record.get("node_start")
            end_node = record.get("node_end")

            # ------------------------------------------------
            # Process nodes
            # ------------------------------------------------

            for node in (start_node, end_node):

                if not isinstance(node, dict):
                    continue

                element_id = node.get("element_id")

                if not element_id:
                    continue

                labels = node.get("labels", [])
                properties = clean_properties(
                    node.get("properties", {})
                )

                search_text = get_text_from_node(
                    {
                        "labels": labels,
                        "properties": properties,
                    }
                )

                relevant = int(
                    is_relevant(
                        {
                            "labels": labels,
                            "properties": properties,
                        }
                    )
                )

                # Count labels
                for label in labels:

                    label_counts[label] = (
                        label_counts.get(label, 0) + 1
                    )

                # Insert/update node.
                # INSERT OR IGNORE prevents repeatedly writing
                # the same node.
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO nodes
                    (
                        element_id,
                        labels,
                        properties,
                        search_text,
                        relevant
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        element_id,
                        json.dumps(labels, ensure_ascii=False),
                        json.dumps(
                            properties,
                            ensure_ascii=False,
                        ),
                        search_text,
                        relevant,
                    ),
                )

                if cursor.rowcount == 1:

                    nodes_new += 1

                    node_record = {
                        "element_id": element_id,
                        "labels": labels,
                        "properties": properties,
                        "search_text": search_text,
                        "relevant": bool(relevant),
                    }

                    nodes_out.write(
                        json.dumps(
                            node_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if relevant:

                        relevant_nodes += 1

                        relevant_out.write(
                            json.dumps(
                                node_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                nodes_seen += 1

            # ------------------------------------------------
            # Process connection between nodes
            # ------------------------------------------------

            if (
                isinstance(start_node, dict)
                and isinstance(end_node, dict)
            ):

                start_id = start_node.get("element_id")
                end_id = end_node.get("element_id")

                if start_id and end_id:

                    start_labels = start_node.get(
                        "labels", []
                    )

                    end_labels = end_node.get(
                        "labels", []
                    )

                    relationship_type, relationship_properties = (
                        extract_relationship_info(record)
                    )

                    cursor.execute(
                        """
                        INSERT INTO relationships
                        (
                            start_id,
                            end_id,
                            start_labels,
                            end_labels,
                            relationship_type,
                            properties
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            start_id,
                            end_id,
                            json.dumps(
                                start_labels,
                                ensure_ascii=False,
                            ),
                            json.dumps(
                                end_labels,
                                ensure_ascii=False,
                            ),
                            relationship_type,
                            json.dumps(
                                relationship_properties,
                                ensure_ascii=False,
                            ),
                        ),
                    )

                    relationships += 1

                    relationship_record = {
                        "start_id": start_id,
                        "end_id": end_id,
                        "start_labels": start_labels,
                        "end_labels": end_labels,
                        "relationship_type": relationship_type,
                        "properties": relationship_properties,
                    }

                    relationships_out.write(
                        json.dumps(
                            relationship_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # ------------------------------------------------
            # Commit periodically
            # ------------------------------------------------

            if records_processed % 1000 == 0:

                conn.commit()

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if records_processed % 10000 == 0:

                elapsed = (
                    datetime.now() - start_time
                ).total_seconds()

                print(
                    f"\rRecords: {records_processed:,} | "
                    f"Unique nodes: {nodes_new:,} | "
                    f"Relationships: {relationships:,} | "
                    f"Relevant nodes: {relevant_nodes:,} | "
                    f"Elapsed: {elapsed / 60:.1f} min",
                    end="",
                    flush=True,
                )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    conn.commit()

    nodes_out.close()
    relevant_out.close()
    relationships_out.close()

    # --------------------------------------------------------
    # Database statistics
    # --------------------------------------------------------

    cursor.execute("SELECT COUNT(*) FROM nodes")
    total_nodes = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM nodes WHERE relevant = 1"
    )
    total_relevant = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM relationships")
    total_relationships = cursor.fetchone()[0]

    conn.close()

    elapsed = (
        datetime.now() - start_time
    ).total_seconds()

    summary = {
        "input_file": INPUT_FILE,
        "input_size_gb": round(file_size_gb, 3),
        "records_processed": records_processed,
        "unique_nodes": total_nodes,
        "relevant_nodes": total_relevant,
        "relationships": total_relationships,
        "embeddings_removed": True,
        "processing_time_seconds": round(elapsed, 2),
        "processing_time_minutes": round(
            elapsed / 60, 2
        ),
        "output_files": {
            "sqlite": DB_FILE,
            "nodes": NODES_FILE,
            "relevant_nodes": RELEVANT_FILE,
            "relationships": RELATIONSHIPS_FILE,
        },
        "label_counts": label_counts,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Records processed : {records_processed:,}")
    print(f"Unique nodes      : {total_nodes:,}")
    print(f"Relevant nodes    : {total_relevant:,}")
    print(f"Relationships     : {total_relationships:,}")
    print(
        f"Processing time   : {elapsed / 60:.2f} minutes"
    )

    print()
    print("Output files:")
    print(f"  SQLite          : {DB_FILE}")
    print(f"  All nodes       : {NODES_FILE}")
    print(f"  Relevant nodes  : {RELEVANT_FILE}")
    print(f"  Relationships   : {RELATIONSHIPS_FILE}")
    print(f"  Summary         : {SUMMARY_FILE}")
    print()

    print("The original 3.38 GB file was NOT modified.")


if __name__ == "__main__":
    main()