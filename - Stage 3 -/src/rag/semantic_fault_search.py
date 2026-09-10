import xml.etree.ElementTree as ET
import pandas as pd
import re
from pathlib import Path

GRAPH_PATH = Path(
    r"data\3GPP_KG\tkg\rel19_3gpp_telecom_kg.graphml"
)

OUTPUT_PATH = Path(
    r"data\3GPP_KG\semantic_fault_search.csv"
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

EDGE_KEYS = {
    "d10": "description",
    "d11": "keywords",
    "d12": "source_id",
    "d13": "file_path",
    "d15": "conditions",
}

SEARCHES = {

    "Doppler Shift": [
        "doppler",
        "frequency shift",
        "doppler spread",
        "time varying channel",
        "time-varying channel",
        "channel variation",
        "channel variation due to mobility",
        "mobility channel",
        "radio propagation",
    ],

    "Jamming": [
        "jamming",
        "jammer",
        "radio jammer",
        "radio jamming",
        "intentional interference",
        "malicious interference",
        "interference attack",
        "interference detection",
        "interference mitigation",
        "radio interference",
    ],

    "Antenna Failure": [
        "antenna failure",
        "antenna fault",
        "antenna malfunction",
        "antenna degradation",
        "antenna performance degradation",
        "antenna problem",
        "antenna abnormal",
        "antenna alarm",
        "antenna",
        "antenna system",
        "antenna characteristics",
    ],

    "Buffer Overflow": [
        "buffer overflow",
        "buffer overrun",
        "buffer exhaustion",
        "buffer full",
        "buffer occupancy",
        "buffer size",
        "queue overflow",
        "queue full",
        "packet buffer",
        "buffer management",
    ],
}


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
        field = NODE_KEYS.get(key)

        if field:
            record[field] = data.text or ""

    nodes[node_id] = record


print(f"Nodes loaded: {len(nodes):,}")


# ---------------------------------------------------------
# Edges
# ---------------------------------------------------------

edges = []

for edge in graph.findall("g:edge", NS):

    record = {
        "source": edge.get("source", ""),
        "target": edge.get("target", ""),
        "description": "",
        "keywords": "",
        "source_id": "",
        "file_path": "",
        "conditions": "",
    }

    for data in edge.findall("g:data", NS):

        key = data.get("key")
        field = EDGE_KEYS.get(key)

        if field:
            record[field] = data.text or ""

    edges.append(record)


print(f"Edges loaded: {len(edges):,}")


# ---------------------------------------------------------
# Build adjacency
# ---------------------------------------------------------

adjacency = {}

for edge in edges:

    adjacency.setdefault(
        edge["source"], []
    ).append(edge)

    adjacency.setdefault(
        edge["target"], []
    ).append(edge)


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

results = []

for fault, terms in SEARCHES.items():

    print()
    print("=" * 80)
    print(f"{fault}")
    print("=" * 80)

    matched_nodes = []

    for node_id, node in nodes.items():

        searchable = " ".join([
            node["entity_id"],
            node["entity_type"],
            node["description"],
        ])

        matched_terms = [
            term
            for term in terms
            if re.search(
                re.escape(term),
                searchable,
                re.IGNORECASE
            )
        ]

        if matched_terms:
            matched_nodes.append(
                (node_id, matched_terms)
            )

    print(
        f"Matching nodes: {len(matched_nodes)}"
    )

    for node_id, matched_terms in matched_nodes:

        node = nodes[node_id]

        node_edges = adjacency.get(
            node_id,
            []
        )

        # Keep entities even if they have no edges
        if not node_edges:

            results.append({
                "fault_query": fault,
                "matched_terms": "; ".join(
                    matched_terms
                ),
                "entity_id": node["entity_id"],
                "entity_type": node["entity_type"],
                "description": node["description"],
                "related_entity": "",
                "related_description": "",
                "edge_description": "",
                "edge_keywords": "",
                "conditions": "",
                "source_id": node["source_id"],
                "file_path": node["file_path"],
            })

            continue

        for edge in node_edges:

            other_id = (
                edge["target"]
                if edge["source"] == node_id
                else edge["source"]
            )

            other = nodes.get(
                other_id,
                {}
            )

            edge_text = " ".join([
                edge["description"],
                edge["keywords"],
                edge["conditions"],
            ])

            all_text = " ".join([
                node["entity_id"],
                node["description"],
                other.get(
                    "entity_id",
                    ""
                ),
                other.get(
                    "description",
                    ""
                ),
                edge_text,
            ])

            relevant_terms = [
                term
                for term in terms
                if re.search(
                    re.escape(term),
                    all_text,
                    re.IGNORECASE
                )
            ]

            results.append({
                "fault_query": fault,
                "matched_terms": "; ".join(
                    relevant_terms
                ),
                "entity_id": node["entity_id"],
                "entity_type": node["entity_type"],
                "description": node["description"],
                "related_entity": other.get(
                    "entity_id",
                    ""
                ),
                "related_description": other.get(
                    "description",
                    ""
                ),
                "edge_description": edge[
                    "description"
                ],
                "edge_keywords": edge[
                    "keywords"
                ],
                "conditions": edge[
                    "conditions"
                ],
                "source_id": (
                    edge["source_id"]
                    or node["source_id"]
                ),
                "file_path": (
                    edge["file_path"]
                    or node["file_path"]
                ),
            })


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df = pd.DataFrame(results)

df = df.drop_duplicates()

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if len(df):

    summary = (
        df.groupby("fault_query")
        .agg(
            evidence_rows=("entity_id", "size"),
            unique_entities=("entity_id", "nunique"),
            unique_related_entities=(
                "related_entity",
                "nunique"
            ),
        )
    )

    print(summary)

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)