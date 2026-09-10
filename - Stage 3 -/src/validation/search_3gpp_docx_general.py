from pathlib import Path
import re


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DOC_PATHS = [
    Path("data/3GPP_Documents/Rel-19/38_series/38133-fy0_s0-11.txt"),
    Path("data/3GPP_Documents/Rel-19/38_series/38133-fy0_sA.1-A.5.txt"),
    Path("data/3GPP_Documents/Rel-19/38_series/38133-fy0_sA.6-A.8.txt"),
    Path("data/3GPP_Documents/Rel-19/38_series/38133-fy0_sB.1-XX.txt"),
]

OUTPUT_DIR = Path(
    "data/3GPP_Documents/Rel-19/38_series/validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Search terms
# ---------------------------------------------------------

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
        "Doppler spread",
        "frequency shift",
        "frequency offset",
        "frequency error",
        "fading",
        "mobility",
        "high mobility",
        "channel variation",
        "time-varying",
        "velocity",
        "speed",
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
# Search function
# ---------------------------------------------------------

def search_term(text, term, context=700):

    results = []

    pattern = re.compile(
        re.escape(term),
        re.IGNORECASE
    )

    for match in pattern.finditer(text):

        start = max(0, match.start() - context)
        end = min(len(text), match.end() + context)

        snippet = text[start:end]

        line_number = (
            text[:match.start()].count("\n") + 1
        )

        results.append({
            "term": term,
            "line": line_number,
            "snippet": snippet.strip()
        })

    return results


# ---------------------------------------------------------
# Process each document
# ---------------------------------------------------------

for DOC_PATH in DOC_PATHS:

    print("\n")
    print("=" * 80)
    print("3GPP DOCUMENT TARGETED VALIDATION")
    print("=" * 80)

    print(f"Loading: {DOC_PATH}")

    if not DOC_PATH.exists():

        print(f"WARNING: File not found: {DOC_PATH}")
        continue

    text = DOC_PATH.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    print(f"Characters: {len(text):,}")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")


    all_results = []


    # -----------------------------------------------------
    # Run searches
    # -----------------------------------------------------

    for group, terms in SEARCH_GROUPS.items():

        print("\n" + "=" * 70)
        print(group)
        print("=" * 70)

        group_results = []

        for term in terms:

            matches = search_term(text, term)

            if matches:

                print(
                    f"{term:35} -> "
                    f"{len(matches)} matches"
                )

                for result in matches:

                    result["fault_group"] = group
                    group_results.append(result)
                    all_results.append(result)

            else:

                print(
                    f"{term:35} -> 0 matches"
                )


    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    output_name = (
        DOC_PATH.stem +
        "_targeted_validation.txt"
    )

    txt_output = OUTPUT_DIR / output_name


    with txt_output.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 80 + "\n")
        f.write("3GPP TARGETED VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Source: {DOC_PATH}\n")
        f.write(f"Characters: {len(text):,}\n\n")


        for group in SEARCH_GROUPS:

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(group + "\n")
            f.write("=" * 80 + "\n\n")

            group_results = [
                r
                for r in all_results
                if r["fault_group"] == group
            ]

            if not group_results:

                f.write("NO MATCHES FOUND\n")
                continue


            for i, result in enumerate(
                group_results,
                1
            ):

                f.write("-" * 80 + "\n")
                f.write(f"Result {i}\n")
                f.write(
                    f"Search term: "
                    f"{result['term']}\n"
                )
                f.write(
                    f"Approximate line: "
                    f"{result['line']}\n\n"
                )
                f.write(
                    result["snippet"]
                )
                f.write("\n\n")


    print(
        f"\nSaved report: {txt_output}"
    )


print("\n")
print("=" * 80)
print("ALL DOCUMENTS VALIDATED")
print("=" * 80)