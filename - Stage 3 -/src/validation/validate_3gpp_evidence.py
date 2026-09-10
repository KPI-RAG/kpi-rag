from pathlib import Path
import re
import csv

# ============================================================
# CONFIGURATION
# ============================================================

DOC_DIR = Path(
    "data/3GPP_Documents/Rel-19/38_series"
)

OUTPUT_DIR = Path(
    "data/3GPP_Documents/Rel-19/38_series/validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "3gpp_evidence_candidates.csv"


# ============================================================
# FAULT -> TECHNICAL SEARCH CONCEPTS
# ============================================================

SEARCH_GROUPS = {

    "Antenna Failure": [
        "antenna failure",
        "antenna fault",
        "antenna malfunction",
        "antenna degradation",
        "antenna impairment",
        "antenna performance",
        "antenna port",
        "antenna gain",
        "antenna pattern",
        "transmit antenna",
        "receive antenna",
        "antenna configuration",
        "radiated performance",
    ],

    "Faulty RF Filters": [
        "RF filter",
        "RF filters",
        "filter failure",
        "filter fault",
        "filter degradation",
        "bandpass filter",
        "out-of-band blocking",
        "receiver blocking",
        "blocking performance",
        "adjacent channel",
        "selectivity",
        "unwanted emissions",
    ],

    "Co-Channel Interference": [
        "co-channel interference",
        "co channel interference",
        "inter-cell interference",
        "intercell interference",
        "interference",
        "interference mitigation",
        "interference rejection",
        "interference cancellation",
        "interfering cell",
        "interfering signal",
    ],

    "Doppler": [
        "Doppler",
        "Doppler shift",
        "Doppler frequency",
        "Doppler spread",
        "frequency shift",
        "frequency offset",
        "fading",
        "time varying channel",
        "time-varying channel",
        "channel variation",
        "high mobility",
    ],

    "Faulty Handover Algorithm": [
        "handover failure",
        "handover failures",
        "handover failure rate",
        "handover failure cause",
        "handover error",
        "handover interruption",
        "handover procedure",
        "handover preparation",
        "handover execution",
        "handover command",
        "handover decision",
        "handover success",
    ],

    "High Network Congestion": [
        "network congestion",
        "traffic congestion",
        "congestion",
        "overload",
        "traffic load",
        "high traffic load",
        "resource congestion",
        "cell load",
        "load balancing",
        "capacity limitation",
    ],

    "Resource Allocation Bugs": [
        "resource allocation",
        "resource assignment",
        "resource allocation failure",
        "resource allocation error",
        "resource shortage",
        "resource exhaustion",
        "radio resource management",
        "radio resource allocation",
        "resource scheduling",
        "resource assignment failure",
    ],

    "Buffer Overflow": [
        "buffer overflow",
        "buffer overrun",
        "buffer exhaustion",
        "buffer full",
        "queue overflow",
        "queue full",
        "memory overflow",
        "memory exhaustion",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def get_context(text, start, end, context=900):
    """
    Extract surrounding text.
    """
    left = max(0, start - context)
    right = min(len(text), end + context)

    return text[left:right].strip()


def find_heading(context):
    """
    Try to identify a nearby numbered clause heading.

    Examples:
        4.1.2
        5.3.1.2
        A.2.3
    """

    patterns = [
        r"\b\d+(?:\.\d+){1,6}\b",
        r"\b[A-Z]\.\d+(?:\.\d+){0,5}\b",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(
            re.findall(pattern, context)
        )

    if candidates:
        return candidates[-1]

    return ""


def classify_candidate(term, context):
    """
    Very conservative heuristic classification.

    This is NOT the final scientific classification.
    It only helps us prioritize manual review.
    """

    context_lower = context.lower()

    direct_indicators = [
        "failure",
        "fault",
        "malfunction",
        "degradation",
        "impairment",
        "error",
        "interference",
        "congestion",
        "overload",
        "shortage",
        "exhaustion",
        "handover failure",
    ]

    indirect_indicators = [
        "performance",
        "requirement",
        "mitigation",
        "blocking",
        "fading",
        "doppler",
        "allocation",
        "assignment",
        "load",
        "capacity",
    ]

    if any(
        indicator in context_lower
        for indicator in direct_indicators
    ):
        return "DIRECT_CANDIDATE"

    if any(
        indicator in context_lower
        for indicator in indirect_indicators
    ):
        return "INDIRECT_CANDIDATE"

    return "CONTEXTUAL"


# ============================================================
# LOAD TXT DOCUMENTS
# ============================================================

txt_files = sorted(
    DOC_DIR.glob("*.txt")
)

if not txt_files:
    raise FileNotFoundError(
        f"No TXT documents found in {DOC_DIR}"
    )

print("=" * 75)
print("3GPP SEMANTIC EVIDENCE CANDIDATE SEARCH")
print("=" * 75)

print(f"Documents found: {len(txt_files)}")

for path in txt_files:
    print(f"  - {path.name}")


# ============================================================
# SEARCH
# ============================================================

results = []

for doc_path in txt_files:

    print("\n" + "=" * 75)
    print(f"DOCUMENT: {doc_path.name}")
    print("=" * 75)

    text = normalize(
        doc_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    )

    for fault, terms in SEARCH_GROUPS.items():

        for term in terms:

            pattern = re.compile(
                re.escape(term),
                re.IGNORECASE
            )

            matches = list(
                pattern.finditer(text)
            )

            for match in matches:

                context = get_context(
                    text,
                    match.start(),
                    match.end()
                )

                line = (
                    text[:match.start()].count("\n")
                    + 1
                )

                clause = find_heading(context)

                evidence_type = classify_candidate(
                    term,
                    context
                )

                results.append({

                    "fault": fault,

                    "document": doc_path.name,

                    "search_term": term,

                    "line": line,

                    "clause_candidate": clause,

                    "evidence_type": evidence_type,

                    "evidence": context,

                })

print("\n" + "=" * 75)
print("SEARCH COMPLETE")
print("=" * 75)

print(f"Candidate evidence records: {len(results):,}")


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "fault",
    "document",
    "search_term",
    "line",
    "clause_candidate",
    "evidence_type",
    "evidence",
]

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


print(f"Saved:")
print(OUTPUT_CSV)