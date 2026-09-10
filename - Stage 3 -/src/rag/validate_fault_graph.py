import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from collections import defaultdict

GRAPH_PATH = Path(
    r"data\3GPP_KG\tkg\rel19_3gpp_telecom_kg.graphml"
)

CANDIDATES_PATH = Path(
    r"data\3GPP_KG\fault_coverage_candidates.csv"
)

OUTPUT_PATH = Path(
    r"data\3GPP_KG\fault_graph_evidence.csv"
)

NS = {
    "g": "http://graphml.graphdrawing.org/xmlns"
}

# GraphML keys
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
    "d18": "id",
    "d17": "release",
    "d16": "source_file",
    "d15": "conditions",
    "d13": "file_path",
    "d12": "source_id",
    "d11": "keywords",
    "d10": "description",
    "d9": "weight",
}

print("Loading GraphML...")
tree = ET.parse(GRAPH_PATH)
root = tree.getroot()
graph = root.find("g:graph", NS)

# ---------------------------------------------------------
# 1. Extract nodes
# ---------------------------------------------------------

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

print(f"Nodes loaded: {len(nodes):,}")

# ---------------------------------------------------------
# 2. Extract edges
# ---------------------------------------------------------

edges_by_node = defaultdict(list)

edge_count = 0

for edge in graph.findall("g:edge", NS):

    source = edge.get("source", "")
    target = edge.get("target", "")

    record = {
        "source_node": source,
        "target_node": target,
        "id": "",
        "release": "",
        "source_file": "",
        "conditions": "",
        "file_path": "",
        "source_id": "",
        "keywords": "",
        "description": "",
        "weight": "",
    }

    for data in edge.findall("g:data", NS):
        key = data.get("key")
        field = EDGE_KEYS.get(key)

        if field:
            record[field] = data.text or ""

    edges_by_node[source].append(record)
    edges_by_node[target].append(record)

    edge_count += 1

print(f"Edges loaded: {edge_count:,}")

# ---------------------------------------------------------
# 3. Load candidate entities
# ---------------------------------------------------------

candidates = pd.read_csv(
    CANDIDATES_PATH
).fillna("")

print(
    f"Candidate rows loaded: {len(candidates):,}"
)

# ---------------------------------------------------------
# 4. Match candidates to graph nodes
# ---------------------------------------------------------

results = []

for _, candidate in candidates.iterrows():

    fault = candidate["telecomts_fault"]

    entity_id = str(candidate.get("entity_id", ""))

    description = str(
        candidate.get("description", "")
    )

    # Find nodes by entity_id
    matched_nodes = []

    for node_id, node in nodes.items():

        if (
            node["entity_id"] == entity_id
            or node_id == entity_id
        ):
            matched_nodes.append(node_id)

    # If no exact entity_id match, try node ID
    if not matched_nodes:

        for node_id, node in nodes.items():

            if node_id == entity_id:
                matched_nodes.append(node_id)

    # Collect neighboring evidence
    for node_id in matched_nodes:

        node = nodes[node_id]

        node_edges = edges_by_node.get(
            node_id,
            []
        )

        # Limit each candidate to its strongest
        # / most informative relationships.
        for edge in node_edges[:20]:

            other_node_id = (
                edge["target_node"]
                if edge["source_node"] == node_id
                else edge["source_node"]
            )

            other = nodes.get(
                other_node_id,
                {}
            )

            results.append({
                "telecomts_fault": fault,
                "candidate_entity": node["entity_id"],
                "candidate_type": node["entity_type"],
                "candidate_description": node["description"],
                "candidate_source_id": node["source_id"],
                "candidate_file_path": node["file_path"],

                "related_entity": other.get(
                    "entity_id", ""
                ),

                "related_type": other.get(
                    "entity_type", ""
                ),

                "related_description": other.get(
                    "description", ""
                ),

                "edge_description": edge[
                    "description"
                ],

                "edge_keywords": edge[
                    "keywords"
                ],

                "edge_conditions": edge[
                    "conditions"
                ],

                "edge_source_id": edge[
                    "source_id"
                ],

                "edge_file_path": edge[
                    "file_path"
                ],

                "edge_weight": edge[
                    "weight"
                ],
            })

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------

evidence = pd.DataFrame(results)

evidence.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print()
print(
    f"Graph evidence rows: {len(evidence):,}"
)

print()
print("Evidence rows by fault:")

print(
    evidence.groupby(
        "telecomts_fault"
    ).size()
    .sort_values(
        ascending=False
    )
)

print()
print(
    f"Saved to: {OUTPUT_PATH}"
)