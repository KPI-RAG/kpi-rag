import xml.etree.ElementTree as ET
import pandas as pd
import re
from pathlib import Path

GRAPH_PATH = Path(
    r"data\3GPP_KG\tkg\rel19_3gpp_telecom_kg.graphml"
)

OUTPUT_PATH = Path(
    r"data\3GPP_KG\doppler_broad_search.csv"
)

NS = {
    "g": "http://graphml.graphdrawing.org/xmlns"
}

NODE_KEYS = {
    "d0": "entity_id",
    "d1": "entity_type",
    "d2": "description",
    "d3": "source_id",
    "d4": "file_path",
}

terms = [
    "doppler",
    "frequency shift",
    "frequency offset",
    "frequency error",
    "frequency variation",
    "frequency variation due to mobility",
    "time varying",
    "time-varying",
    "time variant",
    "time-variant",
    "channel variation",
    "channel variations",
    "channel variation due to mobility",
    "fast fading",
    "fading",
    "mobility",
    "mobile channel",
    "radio channel",
    "propagation channel",
    "channel estimation",
    "channel tracking",
    "channel state",
]

print("Loading GraphML...")
tree = ET.parse(GRAPH_PATH)
root = tree.getroot()
graph = root.find("g:graph", NS)

nodes = {}

for node in graph.findall("g:node", NS):

    node_id = node.get("id", "")

    record = {
        "node_id": node_id,
        "entity_id": "",
        "entity_type": "",
        "description": "",
        "source_id": "",
        "file_path": "",
    }

    for data in node.findall("g:data", NS):

        key = data.get("key")

        if key == "d0":
            record["entity_id"] = data.text or ""
        elif key == "d1":
            record["entity_type"] = data.text or ""
        elif key == "d2":
            record["description"] = data.text or ""
        elif key == "d3":
            record["source_id"] = data.text or ""
        elif key == "d4":
            record["file_path"] = data.text or ""

    nodes[node_id] = record

print(f"Nodes loaded: {len(nodes):,}")

results = []

for node_id, node in nodes.items():

    searchable = " ".join([
        node["entity_id"],
        node["entity_type"],
        node["description"],
    ])

    matched = [
        term
        for term in terms
        if re.search(
            re.escape(term),
            searchable,
            re.IGNORECASE
        )
    ]

    if matched:

        results.append({
            "matched_terms": "; ".join(matched),
            "entity_id": node["entity_id"],
            "entity_type": node["entity_type"],
            "description": node["description"],
            "source_id": node["source_id"],
            "file_path": node["file_path"],
        })

df = pd.DataFrame(results).drop_duplicates()

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(f"Matching evidence rows: {len(df)}")
print(
    f"Unique entities: "
    f"{df['entity_id'].nunique() if len(df) else 0}"
)

print()
print("Matches by term:")

if len(df):
    print(
        df["matched_terms"]
        .str.split("; ")
        .explode()
        .value_counts()
    )

print()
print(f"Saved to: {OUTPUT_PATH}")