import json
import re
import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "oran"
    / "extracted"
    / "oran_relevant_nodes.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "oran"
    / "alignment"
)

OUTPUT_JSON = OUTPUT_DIR / "oran_alignment_evidence.json"
OUTPUT_CSV = OUTPUT_DIR / "oran_alignment_evidence.csv"


# ============================================================
# Authoritative fault taxonomy from the paper
# ============================================================

FAULT_METADATA = {
    "Antenna Failure": {
        "coverage": "Mechanism-level",
        "confidence": "MEDIUM",
    },

    "Buffer Overflow": {
        "coverage": "Mechanism-level",
        "confidence": "MEDIUM",
    },

    "Co-Channel Interference (Mild)": {
        "coverage": "Directly supportable",
        "confidence": "HIGH",
    },

    "Co-Channel Interference (Severe)": {
        "coverage": "Directly supportable",
        "confidence": "HIGH",
    },

    "Doppler Shift": {
        "coverage": "Mechanism-level",
        "confidence": "MEDIUM",
    },

    "Faulty Handover Algorithm (Frequent)": {
        "coverage": "Directly supportable",
        "confidence": "HIGH",
    },

    "Faulty RF Filters": {
        "coverage": "Mechanism-level",
        "confidence": "MEDIUM",
    },

    "High Network Congestion (Static)": {
        "coverage": "Directly supportable",
        "confidence": "MEDIUM",
    },

    "High Network Congestion (Temporal)": {
        "coverage": "Directly supportable",
        "confidence": "MEDIUM",
    },

    "Resource Allocation Bugs": {
        "coverage": "Directly supportable",
        "confidence": "MEDIUM",
    },
}


ANOMALIES = list(FAULT_METADATA.keys())


# ============================================================
# O-RAN concepts relevant to KPI-RAG
# ============================================================

ORAN_CONCEPTS = {

    "O-RU": [
        "o-ru",
        "oru",
        "radio unit",
        "radio processing",
        "rf processing",
        "antenna",
        "radio frequency",
        "rf",
    ],

    "O-DU": [
        "o-du",
        "odu",
        "distributed unit",
        "scheduler",
        "phy",
        "mac",
        "rlc",
        "radio resource",
        "prb",
    ],

    "O-CU": [
        "o-cu",
        "ocu",
        "central unit",
        "cu-cp",
        "cu-up",
        "pdcp",
        "rrc",
    ],

    "Near-RT RIC": [
        "near-rt ric",
        "near rt ric",
        "near-real-time ric",
        "near real-time ric",
        "near-rt",
        "near rt",
    ],

    "Non-RT RIC": [
        "non-rt ric",
        "non rt ric",
        "non-real-time ric",
        "non real-time ric",
    ],

    "SMO": [
        "smo",
        "service management and orchestration",
        "service and management orchestration",
    ],

    "E2": [
        "e2 interface",
        "e2ap",
        "e2sm",
        "e2 node",
    ],

    "A1": [
        "a1 interface",
        "policy-driven guidance",
        "policy driven guidance",
    ],

    "O1": [
        "o1 interface",
        "management interface",
        "performance management",
        "fault management",
    ],

    "O2": [
        "o2 interface",
        "cloud infrastructure",
        "o-cloud",
        "deployment management",
    ],

    "RIC": [
        "ric",
        "ran intelligent controller",
        "ric service",
        "ric policy",
        "ric control",
    ],
}


# ============================================================
# Fault-specific evidence keywords
#
# Each keyword has a weight.
#
# Strong terms = direct mechanism evidence
# Medium terms = supporting mechanism/context
# Weak terms = architectural context
# ============================================================

FAULT_KEYWORDS = {

    "Antenna Failure": {
        "antenna failure": 5,
        "antenna fault": 5,
        "antenna malfunction": 5,
        "antenna": 3,
        "radio unit": 2,
        "o-ru": 2,
        "rf": 1,
        "radio frequency": 2,
        "signal strength": 3,
        "coverage": 2,
        "radiation": 2,
        "transmit": 1,
        "receive": 1,
    },

    "Buffer Overflow": {
        "buffer overflow": 5,
        "buffer": 3,
        "overflow": 5,
        "queue": 3,
        "queue overflow": 5,
        "memory": 2,
        "memory utilization": 3,
        "packet buffering": 3,
        "traffic": 1,
        "congestion": 2,
        "o-du": 2,
        "o-cu": 2,
    },

    "Co-Channel Interference (Mild)": {
        "co-channel interference": 6,
        "co channel interference": 6,
        "interference": 4,
        "inter-cell interference": 5,
        "intra-cell interference": 5,
        "sinr": 4,
        "signal-to-interference": 4,
        "interference level": 4,
        "radio interference": 4,
        "rf interference": 4,
        "frequency reuse": 3,
        "spectrum": 2,
        "radio": 1,
        "o-ru": 2,
    },

    "Co-Channel Interference (Severe)": {
        "co-channel interference": 6,
        "co channel interference": 6,
        "interference": 4,
        "inter-cell interference": 5,
        "intra-cell interference": 5,
        "sinr": 4,
        "signal-to-interference": 4,
        "interference level": 4,
        "strong interference": 6,
        "severe interference": 6,
        "high interference": 5,
        "radio interference": 4,
        "rf interference": 4,
        "spectrum": 2,
        "radio": 1,
        "o-ru": 2,
    },

    "Doppler Shift": {
        "doppler shift": 6,
        "doppler": 5,
        "frequency shift": 5,
        "frequency offset": 3,
        "mobility": 3,
        "high mobility": 4,
        "velocity": 2,
        "moving ue": 3,
        "channel variation": 3,
        "time-varying channel": 4,
        "fast fading": 3,
        "handover": 2,
        "o-du": 1,
        "ric": 1,
    },

    "Faulty Handover Algorithm (Frequent)": {
        "handover": 5,
        "handover algorithm": 6,
        "handover failure": 6,
        "handover failures": 6,
        "frequent handover": 6,
        "frequent handovers": 6,
        "handover decision": 5,
        "mobility management": 4,
        "cell reselection": 4,
        "cell selection": 3,
        "traffic steering": 4,
        "mobility optimization": 4,
        "ric control": 3,
        "ric policy": 3,
        "near-rt ric": 2,
        "e2": 1,
    },

    "Faulty RF Filters": {
        "rf filter": 6,
        "rf filters": 6,
        "filter failure": 6,
        "filter fault": 6,
        "filtering": 3,
        "radio frequency filter": 6,
        "out-of-band": 5,
        "out of band": 5,
        "adjacent channel": 5,
        "adjacent-channel": 5,
        "spectral": 3,
        "spectrum": 2,
        "rf": 2,
        "radio frequency": 2,
        "o-ru": 2,
    },

    "High Network Congestion (Static)": {
        "network congestion": 6,
        "congestion": 5,
        "traffic load": 5,
        "high load": 5,
        "overload": 5,
        "resource utilization": 4,
        "resource utilization efficiency": 4,
        "prb utilization": 5,
        "capacity limitation": 4,
        "network capacity": 3,
        "traffic": 2,
        "throughput": 2,
        "scheduler": 2,
        "radio resource": 3,
        "o-du": 2,
    },

    "High Network Congestion (Temporal)": {
        "network congestion": 6,
        "congestion": 5,
        "traffic load": 5,
        "traffic variation": 5,
        "time-varying traffic": 6,
        "temporal": 3,
        "dynamic load": 5,
        "load variation": 5,
        "traffic burst": 5,
        "bursty traffic": 5,
        "peak traffic": 5,
        "overload": 4,
        "resource utilization": 4,
        "prb utilization": 5,
        "capacity": 3,
        "throughput": 2,
        "scheduler": 2,
        "o-du": 2,
    },

    "Resource Allocation Bugs": {
        "resource allocation": 5,
        "resource allocation failure": 6,
        "resource allocation problem": 6,
        "allocation error": 6,
        "allocation failure": 6,
        "radio resource allocation": 6,
        "radio resource management": 5,
        "rrm": 5,
        "scheduling": 4,
        "scheduler": 3,
        "prb allocation": 6,
        "prb": 2,
        "resource management": 4,
        "resource utilization": 3,
        "ric control": 3,
        "ric policy": 3,
        "o-du": 2,
        "near-rt ric": 2,
    },
}


# ============================================================
# Evidence quality indicators
# ============================================================

DIRECT_MECHANISM_TERMS = [
    "failure",
    "fault",
    "overflow",
    "interference",
    "handover",
    "allocation",
    "filter",
    "congestion",
    "doppler",
    "mobility",
    "scheduling",
    "resource management",
    "traffic steering",
    "signal strength",
]


GENERIC_TERMS = [
    "radio",
    "rf",
    "signal",
    "traffic",
    "channel",
    "performance",
    "memory",
    "network",
]


# ============================================================
# Utility functions
# ============================================================

def normalize(text):
    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_oran_components(text):
    """
    Identify O-RAN components/concepts appearing in a chunk.
    """

    text_normalized = normalize(text)

    matches = []

    for component, keywords in ORAN_CONCEPTS.items():

        for keyword in keywords:

            if keyword in text_normalized:

                matches.append(component)

                break

    return sorted(set(matches))


def classify_evidence_quality(text, matched_keywords):
    """
    Estimate whether the chunk contains actual mechanism-level
    evidence or mostly generic/contextual terminology.
    """

    text_normalized = normalize(text)

    direct_hits = sum(
        1
        for term in DIRECT_MECHANISM_TERMS
        if term in text_normalized
    )

    generic_hits = sum(
        1
        for term in GENERIC_TERMS
        if term in text_normalized
    )

    # Glossary-like chunks often contain many definitions
    glossary_indicators = [
        "abbreviation",
        "key performance indicator",
        "is defined as",
        "means",
        "refers to",
    ]

    glossary_hits = sum(
        1
        for term in glossary_indicators
        if term in text_normalized
    )

    if direct_hits >= 2 and glossary_hits == 0:
        return "mechanism"

    if direct_hits >= 1:
        return "supporting"

    if glossary_hits > 0:
        return "glossary"

    if generic_hits >= 2:
        return "context"

    return "weak"


def score_chunk(text, anomaly):
    """
    Calculate weighted relevance between an O-RAN chunk
    and a paper fault type.
    """

    text_normalized = normalize(text)

    keyword_weights = FAULT_KEYWORDS.get(anomaly, {})

    score = 0
    matched_keywords = []

    for keyword, weight in keyword_weights.items():

        keyword_normalized = normalize(keyword)

        if keyword_normalized in text_normalized:

            score += weight

            matched_keywords.append(keyword)

    if score == 0:
        return 0, [], "weak"

    evidence_quality = classify_evidence_quality(
        text,
        matched_keywords,
    )

    # Penalize glossary-only evidence.
    if evidence_quality == "glossary":
        score = max(1, score - 3)

    # Slightly penalize generic/context-only chunks.
    elif evidence_quality == "context":
        score = max(1, score - 1)

    return score, matched_keywords, evidence_quality


# ============================================================
# Main extraction
# ============================================================

def main():

    print("=" * 70)
    print("O-RAN Static Alignment Evidence Builder")
    print("=" * 70)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if not INPUT_FILE.exists():

        print("ERROR:")
        print(f"Input file does not exist:\n{INPUT_FILE}")

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence = defaultdict(list)

    total_nodes = 0
    chunk_nodes = 0

    print("Reading O-RAN relevant nodes...")
    print()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                node = json.loads(line)

            except json.JSONDecodeError:
                continue

            total_nodes += 1

            labels = node.get("labels", [])

            if "Chunk" not in labels:
                continue

            chunk_nodes += 1

            properties = node.get(
                "properties",
                {}
            )

            text = properties.get(
                "text",
                ""
            )

            if not text:
                continue

            file_name = properties.get(
                "fileName",
                ""
            )

            page_number = properties.get(
                "page_number"
            )

            components = find_oran_components(
                text
            )

            if not components:
                continue

            # ------------------------------------------------
            # Compare against every authoritative fault type
            # ------------------------------------------------

            for anomaly in ANOMALIES:

                score, matched_keywords, evidence_quality = (
                    score_chunk(
                        text,
                        anomaly
                    )
                )

                if score < 2:
                    continue

                evidence[anomaly].append({

                    "score": score,

                    "evidence_quality":
                        evidence_quality,

                    "matched_keywords":
                        matched_keywords,

                    "oran_components":
                        components,

                    "source_document":
                        file_name,

                    "page_number":
                        page_number,

                    "text":
                        text[:3000],

                    "element_id":
                        node.get(
                            "element_id"
                        ),
                })


    # ========================================================
    # Keep strongest evidence
    # ========================================================

    FINAL_RESULTS = []

    MAX_EVIDENCE_PER_ANOMALY = 10

    for anomaly in ANOMALIES:

        candidates = evidence.get(
            anomaly,
            []
        )

        candidates.sort(
            key=lambda x: (
                x["score"],
                1 if x["evidence_quality"] == "mechanism" else 0,
                1 if x["evidence_quality"] == "supporting" else 0,
                len(x["oran_components"]),
            ),
            reverse=True,
        )

        selected = candidates[
            :MAX_EVIDENCE_PER_ANOMALY
        ]

        metadata = FAULT_METADATA[
            anomaly
        ]

        FINAL_RESULTS.append({

            "anomaly":
                anomaly,

            "coverage":
                metadata["coverage"],

            "confidence":
                metadata["confidence"],

            "evidence_count":
                len(candidates),

            "top_evidence":
                selected,

        })


    # ========================================================
    # Save JSON
    # ========================================================

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            FINAL_RESULTS,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # Save CSV
    # ========================================================

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "TelecomTS Fault Type",
            "Coverage",
            "Confidence",
            "Score",
            "Evidence Quality",
            "Matched Keywords",
            "O-RAN Components",
            "Source Document",
            "Page",
            "Evidence",
        ])

        for result in FINAL_RESULTS:

            anomaly = result["anomaly"]

            for item in result["top_evidence"]:

                writer.writerow([

                    anomaly,

                    result["coverage"],

                    result["confidence"],

                    item["score"],

                    item["evidence_quality"],

                    "; ".join(
                        item["matched_keywords"]
                    ),

                    "; ".join(
                        item["oran_components"]
                    ),

                    item["source_document"],

                    item["page_number"],

                    item["text"],
                ])


    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("ALIGNMENT EVIDENCE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Nodes processed : {total_nodes:,}"
    )

    print(
        f"Chunk nodes     : {chunk_nodes:,}"
    )

    print()

    for result in FINAL_RESULTS:

        print(
            f"{result['anomaly']:<42}"
            f" {result['evidence_count']:>6} matches"
        )

    print()
    print("Output files:")
    print(f"  JSON : {OUTPUT_JSON}")
    print(f"  CSV  : {OUTPUT_CSV}")

    print()
    print("Coverage/confidence metadata comes from the paper.")
    print("Evidence is static O-RAN corpus alignment.")
    print("No live Knowledge Graph is created.")

    print("=" * 70)


if __name__ == "__main__":
    main()