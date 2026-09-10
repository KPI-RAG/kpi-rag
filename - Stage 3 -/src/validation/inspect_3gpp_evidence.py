from pathlib import Path
import csv

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(
    "data/3GPP_Documents/Rel-19/38_series"
)

CSV_PATH = (
    BASE_DIR
    / "validation"
    / "3gpp_evidence_candidates.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "validation"
    / "evidence_for_manual_review.txt"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_document(document_name):
    path = BASE_DIR / document_name

    if not path.exists():
        print(f"[WARNING] Missing document: {path}")
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def get_context(text, line_number, radius=8):

    lines = text.splitlines()

    try:
        line_number = int(line_number)
    except (ValueError, TypeError):
        return "[Invalid line number]"

    index = line_number - 1

    if index < 0 or index >= len(lines):
        return "[Line number outside document]"

    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)

    result = []

    for i in range(start, end):
        result.append(
            f"{i + 1:6}: {lines[i]}"
        )

    return "\n".join(result)


# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------

print("=" * 75)
print("3GPP EVIDENCE MANUAL REVIEW EXTRACTOR")
print("=" * 75)

print(f"Loading: {CSV_PATH}")

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}"
    )

with CSV_PATH.open(
    "r",
    encoding="utf-8-sig",
    errors="ignore",
    newline=""
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)


print(f"Candidate records: {len(rows)}")

if not rows:
    raise RuntimeError("CSV contains no records.")


# ---------------------------------------------------------
# Show CSV structure
# ---------------------------------------------------------

print()
print("CSV columns:")
print(reader.fieldnames)

print()
print("Fault values found:")

faults = sorted(
    set(
        row.get("fault", "").strip()
        for row in rows
    )
)

for fault in faults:
    count = sum(
        1
        for row in rows
        if row.get("fault", "").strip() == fault
    )

    print(f"  {fault}: {count}")


# ---------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------

unique = {}

for row in rows:

    fault = row.get("fault", "").strip()
    document = row.get("document", "").strip()
    term = row.get("search_term", "").strip()
    clause = row.get("clause_candidate", "").strip()
    evidence_type = row.get("evidence_type", "").strip()
    line = row.get("line", "").strip()

    key = (
        fault,
        document,
        term,
        clause,
        evidence_type,
        line,
    )

    unique[key] = row


print()
print(f"Unique candidates: {len(unique)}")


# ---------------------------------------------------------
# Cache documents
# ---------------------------------------------------------

documents = {}

for row in unique.values():

    document = row["document"]

    if document not in documents:

        print(
            f"Loading document: {document}"
        )

        documents[document] = load_document(
            document
        )


# ---------------------------------------------------------
# Generate report
# ---------------------------------------------------------

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as out:

    out.write(
        "3GPP EVIDENCE MANUAL REVIEW REPORT\n"
    )

    out.write("=" * 80 + "\n\n")

    current_fault = None

    for row in unique.values():

        fault = row["fault"].strip()

        if fault != current_fault:

            current_fault = fault

            out.write("\n")
            out.write("#" * 80 + "\n")
            out.write(
                f"FAULT: {fault}\n"
            )
            out.write("#" * 80 + "\n\n")

        document = row["document"].strip()
        term = row["search_term"].strip()
        clause = row["clause_candidate"].strip()
        evidence_type = row["evidence_type"].strip()
        line = row["line"].strip()

        text = documents.get(
            document,
            ""
        )

        context = get_context(
            text,
            line,
            radius=8
        )

        out.write("-" * 80 + "\n")

        out.write(
            f"Document       : {document}\n"
        )

        out.write(
            f"Search term    : {term}\n"
        )

        out.write(
            f"Clause         : {clause}\n"
        )

        out.write(
            f"Candidate type : {evidence_type}\n"
        )

        out.write(
            f"Line           : {line}\n\n"
        )

        out.write("CONTEXT:\n")
        out.write(context)
        out.write("\n\n")


# ---------------------------------------------------------
# Done
# ---------------------------------------------------------

print()
print("=" * 75)
print("REVIEW REPORT CREATED")
print("=" * 75)

print(f"Output:")
print(OUTPUT_PATH)