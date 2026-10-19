import pandas as pd
from pathlib import Path

INPUT = Path(r"data\3GPP_KG\fault_coverage_candidates.csv")
OUTPUT = Path(r"data\3GPP_KG\fault_top_candidates.csv")

df = pd.read_csv(INPUT).fillna("")

# Terms that indicate stronger semantic relevance.
priority_terms = {
    "Antenna Failure": [
        "antenna failure",
        "antenna fault",
        "antenna malfunction",
        "antenna",
    ],

    "Buffer Overflow": [
        "buffer overflow",
        "buffer",
        "overflow",
    ],

    "Co-Channel Interference": [
        "co-channel interference",
        "cochannel interference",
        "interference",
    ],

    "Doppler Shift": [
        "doppler shift",
        "doppler",
        "frequency shift",
    ],

    "Faulty Handover Algorithm": [
        "handover algorithm",
        "handover failure",
        "handover procedure",
        "handover",
    ],

    "Faulty RF Filters": [
        "rf filter",
        "radio frequency filter",
        "filter",
    ],

    "High Network Congestion": [
        "network congestion",
        "congestion",
        "overload",
    ],

    "Resource Allocation Bugs": [
        "resource allocation",
        "radio resource",
        "resource scheduling",
        "resource management",
    ],

    "Jamming": [
        "radio jamming",
        "jamming",
        "intentional interference",
        "interference",
    ],
}


def score_row(row):

    fault = row["telecomts_fault"]

    text = " ".join(
        str(row.get(column, ""))
        for column in [
            "entity_id",
            "entity_type",
            "description",
            "matched_terms",
        ]
    ).lower()

    score = 0
    matched = []

    for term in priority_terms.get(fault, []):
        if term.lower() in text:
            matched.append(term)

            # Longer, more specific phrases receive more weight.
            score += len(term.split()) * 2

    return pd.Series({
        "candidate_score": score,
        "priority_matches": "; ".join(matched),
    })


scores = df.apply(score_row, axis=1)

df = pd.concat([df, scores], axis=1)

df = df.sort_values(
    ["telecomts_fault", "candidate_score"],
    ascending=[True, False],
)

# Keep top 20 candidates per fault.
top = (
    df.groupby("telecomts_fault", group_keys=False)
      .head(20)
)

top.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8",
)

print("Top candidates:")
print()

for fault, group in top.groupby("telecomts_fault"):
    print(f"\n=== {fault} ===")

    for _, row in group.head(10).iterrows():
        print(
            f"[{row['candidate_score']}] "
            f"{row['entity_id']} | "
            f"{row['entity_type']} | "
            f"{row['description'][:180]}"
        )

print()
print(f"Saved to: {OUTPUT}")