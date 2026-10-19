import pandas as pd
from pathlib import Path

INPUT = Path(
    r"data\3GPP_KG\fault_graph_evidence.csv"
)

OUTPUT = Path(
    r"data\3GPP_KG\fault_evidence_review.csv"
)

df = pd.read_csv(INPUT).fillna("")

faults = [
    "Co-Channel Interference",
    "Faulty Handover Algorithm",
    "High Network Congestion",
    "Resource Allocation Bugs",
    "Antenna Failure",
    "Buffer Overflow",
    "Faulty RF Filters",
]

# ---------------------------------------------------------
# Create a relevance score
# ---------------------------------------------------------

def score(row):

    text = " ".join([
        str(row["candidate_entity"]),
        str(row["candidate_description"]),
        str(row["related_entity"]),
        str(row["related_description"]),
        str(row["edge_description"]),
        str(row["edge_keywords"]),
        str(row["edge_conditions"]),
    ]).lower()

    score = 0

    strong_terms = [
        "failure",
        "fault",
        "interference",
        "handover",
        "congestion",
        "overload",
        "resource allocation",
        "radio resource",
        "antenna",
        "buffer",
        "filter",
    ]

    for term in strong_terms:
        if term in text:
            score += 1

    # Give additional weight to explicit edge descriptions
    if row["edge_description"].strip():
        score += 2

    if row["edge_keywords"].strip():
        score += 2

    if row["edge_conditions"].strip():
        score += 2

    return score


df["relevance_score"] = df.apply(
    score,
    axis=1
)

df = df[
    df["telecomts_fault"].isin(faults)
]

# Remove exact duplicate relationships
df = df.drop_duplicates(
    subset=[
        "telecomts_fault",
        "candidate_entity",
        "related_entity",
        "edge_description",
        "edge_keywords",
    ]
)

# Sort strongest evidence first
df = df.sort_values(
    [
        "telecomts_fault",
        "relevance_score",
    ],
    ascending=[
        True,
        False,
    ],
)

# Keep top 15 per fault
review = (
    df.groupby(
        "telecomts_fault",
        group_keys=False
    )
    .head(15)
)

review.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

# ---------------------------------------------------------
# Print readable output
# ---------------------------------------------------------

for fault, group in review.groupby(
    "telecomts_fault"
):

    print("\n" + "=" * 80)
    print(fault)
    print("=" * 80)

    for _, row in group.iterrows():

        print(
            f"\nScore: {row['relevance_score']}"
        )

        print(
            f"Candidate: "
            f"{row['candidate_entity']}"
        )

        print(
            f"Description: "
            f"{row['candidate_description'][:250]}"
        )

        print(
            f"Related: "
            f"{row['related_entity']}"
        )

        print(
            f"Related description: "
            f"{row['related_description'][:250]}"
        )

        print(
            f"Edge description: "
            f"{row['edge_description'][:250]}"
        )

        print(
            f"Keywords: "
            f"{row['edge_keywords'][:200]}"
        )

        print(
            f"Conditions: "
            f"{row['edge_conditions'][:200]}"
        )

        print(
            f"Source: "
            f"{row['edge_file_path']}"
        )

print(
    f"\nSaved review file to: {OUTPUT}"
)