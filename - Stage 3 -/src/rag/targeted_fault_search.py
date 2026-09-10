import xml.etree.ElementTree as ET
import pandas as pd
import re
from pathlib import Path

GRAPH_PATH = Path(
    r"data\3GPP_KG\tkg\rel19_3gpp_telecom_kg.graphml"
)

OUTPUT_PATH = Path(
    r"data\3GPP_KG\targeted_fault_search.csv"
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
    "d6": "source_file",
    "d7": "release",
}

EDGE_KEYS = {
    "d10": "description",
    "d11": "keywords",
    "d12": "source_id",
    "d13": "file_path",
    "d15": "conditions",
}

# ---------------------------------------------------------
# Targeted searches
# ---------------------------------------------------------

queries = {

    "Doppler Shift": [
        "doppler",
        "doppler shift",
        "doppler effect",
        "doppler frequency",
        "doppler spread",
        "frequency shift",
    ],

    "Jamming": [
        "jamming",
        "jammer",
        "radio jamming",
        "jamming attack",
        "intentional interference",
        "malicious interference",
    ],

    "Faulty RF Filters": [
        "rf filter",
        "radio frequency filter",
        "rf filtering",
        "band-pass filter",
        "bandpass filter",
        "low-pass filter",
        "high-pass filter",
        "rf front-end",
        "filter failure",
        "filter malfunction",
    ],

    "Antenna Failure": [
        "antenna failure",
        "antenna fault",
        "antenna malfunction",
        "antenna degradation",
    ],

    "Buffer Overflow": [
        "buffer overflow",
        "buffer overrun",
        "buffer exhaustion",
        "buffer overflow failure",
    ],

    "Handover Failure": [
        "handover failure",
        "handover failures",
        "handover error",
        "handover failure cause",
        "handover problem",
    ],

    "Interference Failure": [
        "interference failure",
        "interference problem",
        "interference mitigation",
        "interference management",
    ],

    "Congestion / Overload": [
        "network congestion",
        "congestion control",
        "congestion management",
        "overload condition",
        "system overload",
    ],

    "Resource Allocation Failure": [
        "resource allocation failure",
        "resource allocation problem",
        "resource allocation error",
        "resource allocation",
        "resource management failure",
    ],
}


print("Loading GraphML...")
tree = ET.parse(GRAPH_PATH)
root = tree.getroot()
graph = root.find("g:graph", NS)

print("Extracting nodes...")

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
        "source_file": "",
        "release": "",
    }

    for data in node.findall("g:data", NS):

        key = data.get("key")
        field = NODE_KEYS.get(key)

        if field:
            record[field] = data.text or ""

    nodes[node_id] = record


print(f"Nodes: {len(nodes):,}")

# ---------------------------------------------------------
# Extract edges
# ---------------------------------------------------------

print("Extracting edges...")

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

print(f"Edges: {len(edges):,}")

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

for fault, terms in queries.items():

    print()
    print("=" * 70)
    print(f"{fault}")
    print("=" * 70)

    pattern = re.compile(
        "|".join(
            re.escape(term)
            for term in terms
        ),
        re.IGNORECASE
    )

    matched_nodes = []

    # Search node fields
    for node_id, node in nodes.items():

        searchable = " ".join([
            node["entity_id"],
            node["entity_type"],
            node["description"],
        ])

        if pattern.search(searchable):
            matched_nodes.append(node_id)

    print(
        f"Matching nodes: {len(matched_nodes)}"
    )

    # Get graph context
    for node_id in matched_nodes:

        node = nodes[node_id]

        node_edges = adjacency.get(
            node_id,
            []
        )

        # If node has no edges, still preserve it
        if not node_edges:

            results.append({
                "fault_query": fault,
                "matched_term": "",
                "entity_id": node["entity_id"],
                "entity_type": node["entity_type"],
                "description": node["description"],
                "related_entity": "",
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

            # Identify which query terms occur
            matched_terms = [
                term
                for term in terms
                if re.search(
                    re.escape(term),
                    (
                        node["entity_id"]
                        + " "
                        + node["description"]
                        + " "
                        + edge_text
                    ),
                    re.IGNORECASE,
                )
            ]

            results.append({
                "fault_query": fault,
                "matched_term": "; ".join(
                    matched_terms
                ),
                "entity_id": node["entity_id"],
                "entity_type": node["entity_type"],
                "description": node["description"],
                "related_entity": other.get(
                    "entity_id",
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

# Remove duplicates
df = df.drop_duplicates()

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

summary = (
    df.groupby("fault_query")
    .agg(
        rows=("entity_id", "size"),
        entities=("entity_id", "nunique"),
        related_entities=(
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