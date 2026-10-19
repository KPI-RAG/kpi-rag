import xml.etree.ElementTree as ET
import csv
from pathlib import Path

GRAPH_PATH = Path(
    r"data\3GPP_KG\tkg\rel19_3gpp_telecom_kg.graphml"
)

OUTPUT_PATH = Path(
    r"data\3GPP_KG\gsma_entities.csv"
)

NS = {
    "g": "http://graphml.graphdrawing.org/xmlns"
}

KEY_MAP = {
    "d0": "entity_id",
    "d1": "entity_type",
    "d2": "description",
    "d3": "source_id",
    "d4": "file_path",
    "d5": "created_at",
    "d6": "source_file",
    "d7": "release",
    "d8": "raw_entity_id",
}

tree = ET.parse(GRAPH_PATH)
root = tree.getroot()

graph = root.find("g:graph", NS)

records = []

for node in graph.findall("g:node", NS):

    record = {
        "node_id": node.get("id", "")
    }

    for data in node.findall("g:data", NS):
        key = data.get("key")
        field = KEY_MAP.get(key)

        if field:
            record[field] = data.text or ""

    records.append(record)

print("Number of KG entities:", len(records))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "node_id",
            "entity_id",
            "entity_type",
            "description",
            "source_id",
            "file_path",
            "created_at",
            "source_file",
            "release",
            "raw_entity_id",
        ],
    )

    writer.writeheader()
    writer.writerows(records)

print(f"Saved to: {OUTPUT_PATH}")