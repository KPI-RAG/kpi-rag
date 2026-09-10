from pathlib import Path
import re

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DOC_PATH = Path(
    "data/3GPP_Documents/Rel-19/38_series/38141-1-j00.txt"
)

OUTPUT_DIR = Path("data/3GPP_Documents/Rel-19/38_series/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Terms derived from the gaps we observed in the KG.
SEARCH_GROUPS = {
    "Antenna Failure": [
        "antenna failure",
        "antenna fault",
        "antenna malfunction",
        "antenna degradation",
        "antenna damage",
        "antenna failure alarm",
        "antenna performance",
        "antenna system",
        "antenna",
    ],

    "Faulty RF Filters": [
        "RF filter",
        "RF filters",
        "filter failure",
        "filter fault",
        "filter malfunction",
        "filter degradation",
        "bandpass filter",
        "blocking",
        "out-of-band blocking",
        "receiver blocking",
        "interference cancellation",
    ],

    "Doppler / Mobility": [
        "Doppler",
        "Doppler shift",
        "Doppler frequency",
        "frequency shift",
        "frequency offset",
        "frequency error",
        "fading",
        "mobility",
        "high mobility",
        "channel variation",
    ],

    "Buffer Overflow": [
        "buffer overflow",
        "buffer overrun",
        "buffer exhaustion",
        "buffer full",
        "buffer capacity",
        "overflow",
        "memory overflow",
        "queue overflow",
    ],

    "Congestion / Overload": [
        "congestion",
        "overload",
        "network congestion",
        "traffic congestion",
        "resource congestion",
        "resource overload",
        "high traffic load",
    ],

    "Resource Allocation": [
        "resource allocation",
        "resource assignment",
        "resource allocation failure",
        "resource allocation error",
        "resource shortage",
        "resource exhaustion",
        "radio resource management",
    ],

    "Handover": [
        "handover failure",
        "handover failure rate",
        "handover failure cause",
        "handover error",
        "handover interruption",
        "handover procedure",
        "handover",
    ],

    "Interference": [
        "co-channel interference",
        "co channel interference",
        "inter-cell interference",
        "interference",
        "interference mitigation",
        "interference cancellation",
        "interference rejection",
    ],
}


# ---------------------------------------------------------
# Load document
# ---------------------------------------------------------

print("=" * 70)
print("3GPP DOCUMENT TARGETED VALIDATION")
print("=" * 70)

print(f"Loading: {DOC_PATH}")

if not DOC_PATH.exists():
    raise FileNotFoundError(f"Document not found: {DOC_PATH}")

text = DOC_PATH.read_text(encoding="utf-8", errors="ignore")

print(f"Characters: {len(text):,}")

# Normalize line endings
text = text.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------
# Search with context
# ---------------------------------------------------------

def search_term(text, term, context=700):
    """
    Return occurrences of a term with surrounding context.
    """
    results = []

    pattern = re.compile(re.escape(term), re.IGNORECASE)

    for match in pattern.finditer(text):
        start = max(0, match.start() - context)
        end = min(len(text), match.end() + context)

        snippet = text[start:end]

        # Find approximate paragraph/line information
        line_number = text[:match.start()].count("\n") + 1

        results.append({
            "term": term,
            "line": line_number,
            "snippet": snippet.strip()
        })

    return results


# ---------------------------------------------------------
# Run searches
# ---------------------------------------------------------

all_results = []

for group, terms in SEARCH_GROUPS.items():

    print("\n" + "=" * 70)
    print(group)
    print("=" * 70)

    group_results = []

    for term in terms:

        matches = search_term(text, term)

        if matches:
            print(f"{term:35} -> {len(matches)} matches")

            for result in matches:
                result["fault_group"] = group
                group_results.append(result)
                all_results.append(result)

        else:
            print(f"{term:35} -> 0 matches")


# ---------------------------------------------------------
# Save TXT report
# ---------------------------------------------------------

txt_output = OUTPUT_DIR / "targeted_3gpp_validation.txt"

with txt_output.open("w", encoding="utf-8") as f:

    for group in SEARCH_GROUPS:

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(group + "\n")
        f.write("=" * 80 + "\n\n")

        group_results = [
            r for r in all_results
            if r["fault_group"] == group
        ]

        if not group_results:
            f.write("NO MATCHES FOUND\n")
            continue

        for i, result in enumerate(group_results, 1):

            f.write("-" * 80 + "\n")
            f.write(f"Result {i}\n")
            f.write(f"Search term: {result['term']}\n")
            f.write(f"Approximate line: {result['line']}\n")
            f.write("\n")
            f.write(result["snippet"])
            f.write("\n\n")


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print(f"Total evidence matches: {len(all_results):,}")
print(f"Saved to: {txt_output}")