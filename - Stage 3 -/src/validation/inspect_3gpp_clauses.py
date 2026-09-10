from pathlib import Path
import re


# ============================================================
# CONFIGURATION
# ============================================================

DOC_DIR = Path("data/3GPP_Documents/Rel-19/38_series")

TARGETS = {
    "Faulty RF Filters": [
        ("38141-1-j00.txt", "D.4.3"),
        ("38141-1-j00.txt", "D.4.4"),
        ("38141-1-j00.txt", "D.6.1"),
        ("38141-1-j00.txt", "D.3.1"),
        ("38141-1-j00.txt", "D.4.5"),
        ("38141-1-j00.txt", "D.4.6"),
        ("38141-1-j00.txt", "D.5.2"),
        ("38141-1-j00.txt", "D.3.3"),
        ("38141-1-j00.txt", "D.3.2"),
        ("38141-1-j00.txt", "D.3.4"),
    ]
}


# Maximum number of lines shown in PowerShell.
# The complete clause is still written to the review file.
PREVIEW_LINES = 5


# ============================================================
# CLAUSE EXTRACTION
# ============================================================

def normalize_line(line):
    """
    Normalize whitespace for matching while preserving
    the original line for output.
    """
    return re.sub(r"\s+", " ", line.strip())


def clause_pattern(clause):
    """
    Create an exact clause-heading pattern.

    Important:
        D.5.2 must NOT match D.5.2A.
    """
    return re.compile(
        rf"^\s*\[P\d+\]\s+{re.escape(clause)}(?:\s+|$)"
    )


def extract_clause(text, clause):
    """
    Extract the actual complete clause from the document body.

    3GPP text files commonly contain the clause heading twice:
        1. Once in the Table of Contents.
        2. Once at the actual location of the clause.

    Therefore, we use the LAST exact occurrence of the requested
    clause heading as the starting point.

    Example:

        [P706]  D.4.3 ...        <- Table of Contents

        ...

        [P4462] D.4.3 ...        <- Actual clause

    The extraction then stops at the next actual clause heading
    of the same or higher structural level.
    """

    lines = text.splitlines()

    target_regex = clause_pattern(clause)

    # --------------------------------------------------------
    # Find ALL exact occurrences of the requested clause
    # --------------------------------------------------------

    matches = []

    for i, line in enumerate(lines):
        if target_regex.search(line):
            matches.append(i)

    if not matches:
        return None

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The first occurrence is normally in the TOC.
    # The last occurrence is the actual clause in the body.
    # --------------------------------------------------------

    start = matches[-1]

    # --------------------------------------------------------
    # Determine structural depth.
    #
    # D.4       -> depth 2
    # D.4.3     -> depth 3
    # D.4.3.1   -> depth 4
    #
    # We count the number of components rather than treating
    # the initial D as a component of the hierarchy.
    # --------------------------------------------------------

    depth = len(clause.split("."))

    result = [lines[start]]

    # --------------------------------------------------------
    # Generic clause heading
    #
    # Matches:
    #
    # [P4465] D.4.4
    # [P4472] D.4.5
    # [P4475] D.4.6
    # [P4484] D.5.2
    #
    # Does NOT match:
    #
    # Figure D.4.4-1
    # --------------------------------------------------------

    generic_clause_regex = re.compile(
        r"^\s*\[P\d+\]\s+"
        r"([A-Z]\.\d+(?:\.\d+)*)"
        r"(?:\s+|$)"
    )

    next_clause = None
    next_index = None

    # --------------------------------------------------------
    # Search ONLY AFTER the actual clause heading.
    # --------------------------------------------------------

    for i in range(start + 1, len(lines)):

        line = lines[i]

        match = generic_clause_regex.match(line)

        if not match:
            continue

        candidate = match.group(1)

        candidate_parts = candidate.split(".")

        # ----------------------------------------------------
        # Candidate must be same level or higher.
        #
        # Example for D.4.3:
        #
        # D.4.3.1 -> deeper -> continue
        # D.4.3.2 -> deeper -> continue
        # D.4.4   -> same level -> STOP
        # D.5     -> higher level -> STOP
        # ----------------------------------------------------

        if len(candidate_parts) <= depth:

            next_clause = candidate
            next_index = i
            break

    # --------------------------------------------------------
    # Special handling for D.5.2A
    #
    # D.5.2A is not matched by the purely numeric structure
    # above, but it must terminate D.5.2.
    # --------------------------------------------------------

    if clause == "D.5.2":

        special_regex = re.compile(
            r"^\s*\[P\d+\]\s+D\.5\.2A(?:\s+|$)"
        )

        for i in range(start + 1, len(lines)):

            if special_regex.search(lines[i]):

                if next_index is None or i < next_index:
                    next_clause = "D.5.2A"
                    next_index = i

                break

    # --------------------------------------------------------
    # Extract the actual clause
    # --------------------------------------------------------

    if next_index is not None:

        result.extend(
            lines[start + 1:next_index]
        )

    else:

        result.extend(
            lines[start + 1:]
        )

    return {
        "clause": clause,

        # Convert Python 0-based index to human-readable line.
        "start_line": start + 1,

        # The boundary line itself belongs to the next clause,
        # so keep the existing convention here.
        "end_line": (
            next_index
            if next_index is not None
            else len(lines)
        ),

        "next_clause": next_clause,

        "text": "\n".join(result).strip(),
    }
# ============================================================
# TERMINAL PREVIEW
# ============================================================

def print_preview(evidence):
    """
    Print a short preview of the extracted clause to PowerShell.

    The complete clause is still written to the review file.
    """

    text = evidence["text"]
    lines = text.splitlines()

    print(f"Extracted characters: {len(text):,}")
    print(f"Extracted lines:      {len(lines):,}")

    # Python source-file line numbers are 1-based here.
    print(
        f"Source text lines:    "
        f"{evidence['start_line']} -> {evidence['end_line']}"
    )

    if evidence["next_clause"]:
        print(
            f"Stopped before clause: {evidence['next_clause']}"
        )
    else:
        print("Stopped at end of document")

    print("\nPreview:")

    preview = lines[:PREVIEW_LINES]

    for line in preview:
        print(f"  {line}")

    if len(lines) > PREVIEW_LINES:
        remaining = len(lines) - PREVIEW_LINES
        print(
            f"  ... [{remaining:,} more lines "
            f"written to review file]"
        )

    print(
        "\nNote: [P####] values are 3GPP paragraph IDs, "
        "not source-file line numbers."
    )
# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("3GPP TARGETED CLAUSE INSPECTION")
print("=" * 75)

output_dir = DOC_DIR / "validation"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "rf_filter_clause_review.txt"


total_targets = 0
successful_targets = 0
failed_targets = 0


with output_file.open(
    "w",
    encoding="utf-8"
) as out:

    for fault, targets in TARGETS.items():

        out.write("\n")
        out.write("=" * 75 + "\n")
        out.write(fault + "\n")
        out.write("=" * 75 + "\n\n")

        print(f"\n{fault}")

        for filename, clause in targets:

            total_targets += 1

            path = DOC_DIR / filename

            print("\n" + "-" * 75)
            print(f"Document: {filename}")
            print(f"Clause:   {clause}")

            out.write("-" * 75 + "\n")
            out.write(f"Document: {filename}\n")
            out.write(f"Clause: {clause}\n\n")

            # ------------------------------------------------
            # File existence
            # ------------------------------------------------

            if not path.exists():

                print(f"[MISSING] {filename}")

                out.write(
                    "STATUS: FILE NOT FOUND\n\n"
                )

                failed_targets += 1
                continue

            # ------------------------------------------------
            # Read document
            # ------------------------------------------------

            try:

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

            except Exception as exc:

                print(
                    f"[ERROR] Could not read {filename}: {exc}"
                )

                out.write(
                    f"STATUS: READ ERROR\n"
                    f"ERROR: {exc}\n\n"
                )

                failed_targets += 1
                continue

            # ------------------------------------------------
            # Extract
            # ------------------------------------------------

            evidence = extract_clause(
                text,
                clause
            )

            if evidence is None:

                print("STATUS: CLAUSE NOT FOUND")

                out.write(
                    "STATUS: CLAUSE NOT FOUND\n\n"
                )

                failed_targets += 1
                continue

            # ------------------------------------------------
            # Successful extraction
            # ------------------------------------------------

            successful_targets += 1

            clause_text = evidence["text"]
            clause_lines = clause_text.splitlines()

            print("STATUS: FOUND")

            print_preview(evidence)

            # ------------------------------------------------
            # Write metadata
            # ------------------------------------------------

            out.write(
                "STATUS: FOUND\n"
            )

            out.write(
                f"Start source line: "
                f"{evidence['start_line']}\n"
            )

            out.write(
                f"End source line: "
                f"{evidence['end_line']}\n"
            )

            out.write(
                f"Extracted characters: "
                f"{len(clause_text):,}\n"
            )

            out.write(
                f"Extracted lines: "
                f"{len(clause_lines):,}\n"
            )

            out.write(
                f"Next clause: "
                f"{evidence['next_clause']}\n\n"
            )

            out.write(
                "FULL CLAUSE CONTENT\n"
            )

            out.write(
                "-" * 75 + "\n"
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Write the COMPLETE clause.
            # There is NO [:1000] truncation here.
            # ------------------------------------------------

            out.write(clause_text)

            out.write("\n\n")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("REVIEW FILE CREATED")
print("=" * 75)

print(f"Output: {output_file}")

print(
    f"\nTargets:    {total_targets}"
)

print(
    f"Extracted:  {successful_targets}"
)

print(
    f"Failed:     {failed_targets}"
)

if failed_targets == 0:

    print(
        "\nSTATUS: ALL TARGET CLAUSES FOUND"
    )

else:

    print(
        "\nSTATUS: SOME TARGETS FAILED"
    )

print(
    "\nThe terminal shows only a short preview."
)

print(
    "The review file contains the COMPLETE extracted clauses."
)