import pandas as pd
import re
from pathlib import Path

INPUT = Path(r"data\3GPP_KG\gsma_entities.csv")
OUTPUT = Path(r"data\3GPP_KG\fault_coverage_candidates.csv")

df = pd.read_csv(INPUT).fillna("")

# Search terms deliberately include both the TelecomTS terminology
# and likely telecom/3GPP terminology.
fault_queries = {
    "Antenna Failure": [
        "antenna",
        "antenna failure",
        "antenna fault",
        "antenna malfunction",
    ],

    "Buffer Overflow": [
        "buffer",
        "buffer overflow",
        "overflow",
        "buffer management",
    ],

    "Co-Channel Interference": [
        "co-channel interference",
        "cochannel interference",
        "co-channel",
        "interference",
    ],

    "Doppler Shift": [
        "doppler",
        "doppler shift",
        "frequency shift",
    ],

    "Faulty Handover Algorithm": [
        "handover",
        "handover failure",
        "handover procedure",
        "handover algorithm",
        "mobility",
    ],

    "Faulty RF Filters": [
        "rf filter",
        "radio frequency filter",
        "filter",
        "filtering",
    ],

    "High Network Congestion": [
        "congestion",
        "network congestion",
        "traffic congestion",
        "overload",
    ],

    "Resource Allocation Bugs": [
        "resource allocation",
        "resource management",
        "radio resource",
        "resource scheduling",
        "allocation",
    ],
}

search_columns = [
    "entity_id",
    "entity_type",
    "description",
]

results = []

for fault, terms in fault_queries.items():

    # Build one regex from all search terms
    pattern = "|".join(
        re.escape(term)
        for term in terms
    )

    mask = pd.Series(False, index=df.index)

    for column in search_columns:
        mask |= df[column].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )

    matches = df[mask].copy()

    matches.insert(0, "telecomts_fault", fault)

    # Calculate which terms matched
    def matched_terms(row):
        combined = " ".join(
            str(row[col])
            for col in search_columns
        ).lower()

        return "; ".join(
            term for term in terms
            if term.lower() in combined
        )

    matches["matched_terms"] = matches.apply(
        matched_terms,
        axis=1
    )

    results.append(matches)

results_df = pd.concat(
    results,
    ignore_index=True
)

# Remove exact duplicate rows
results_df = results_df.drop_duplicates()

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print("Total candidate matches:", len(results_df))
print()
print("Matches by TelecomTS fault:")
print(
    results_df.groupby("telecomts_fault")
    .size()
    .sort_values(ascending=False)
)

print()
print(f"Saved to: {OUTPUT}")