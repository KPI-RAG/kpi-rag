from pathlib import Path
import re


SOURCE = Path(
    "data/3GPP_Documents/Rel-19/38_series/38141-1-j00.txt"
)

OUTPUT = Path(
    "data/3GPP_Documents/Rel-19/38_series/validation/"
    "receiver_requirements_review.txt"
)


TARGETS = [
    "7.4.1",
    "7.4.2",
    "7.5",
    "7.6",
    "7.7",
    "7.8",
]


def normalize_clause(clause: str) -> str:
    return clause.strip()


def clause_pattern(clause: str):
    """
    Match the exact requested clause number.

    Examples:
        7.5      -> [Pxxxx] 7.5 ...
        7.4.1    -> [Pxxxx] 7.4.1 ...
        7.4.2    -> [Pxxxx] 7.4.2 ...
    """

    escaped = re.escape(clause)

    return re.compile(
        rf"^\[P\d+\]\s+{escaped}\s+"
    )


def next_clause_pattern(clause: str):
    """
    Match a clause at the same hierarchical level or above.

    For 7.4.1:
        7.4.2  -> stop
        7.5    -> stop
        7.4.1.1 -> do NOT stop

    For 7.5:
        7.6 -> stop
        7.5.1 -> do NOT stop
    """

    parts = clause.split(".")

    if len(parts) == 2:
        # 7.5 -> next 7.x clause
        return re.compile(
            rf"^\[P\d+\]\s+7\.(?!{re.escape(parts[1])}\b)"
        )

    if len(parts) == 3:
        major, middle, minor = parts

        return re.compile(
            rf"^\[P\d+\]\s+"
            rf"{re.escape(major)}\."
            rf"(?:{re.escape(middle)}\.)?"
            rf"\d+\s+"
        )

    raise ValueError(f"Unsupported clause format: {clause}")


def extract_clause(lines, target):
    start_pattern = clause_pattern(target)

    start_index = None

    for i, line in enumerate(lines):
        if start_pattern.search(line):
            start_index = i
            break

    if start_index is None:
        return None

    # For major sections such as 7.5,
    # stop when the next 7.x section appears.
    if len(target.split(".")) == 2:
        target_number = target.split(".")[1]

        # Example:
        # target = 7.8
        #
        # Stop at:
        #   7.9
        #   7.10
        #   8.1
        #
        # Do NOT stop at:
        #   7.8.1
        #   7.8.2
        #   7.8.5
        top_level_pattern = re.compile(
            r"^\[P\d+\]\s+(\d+)\.(\d+)\s+"
        )

        end_index = len(lines)

        for i in range(start_index + 1, len(lines)):
            match = top_level_pattern.match(lines[i])

            if not match:
                continue

            major = match.group(1)
            section = match.group(2)

            # We are currently inside Clause 7.x.
            # Stop at the next 7.x section.
            if major == "7" and section != target_number:
                end_index = i
                break

            # Also stop if the document moves to another
            # major clause, e.g. 8.x.
            if major != "7":
                end_index = i
                break

    else:
        # For 7.4.1 / 7.4.2:
        # stop at the next sibling or parent section.
        target_parts = target.split(".")
        major = target_parts[0]
        section = target_parts[1]
        subsection = target_parts[2]

        end_index = len(lines)

        for i in range(start_index + 1, len(lines)):
            line = lines[i]

            match = re.match(
                rf"^\[P\d+\]\s+({re.escape(major)}\.\d+(?:\.\d+)?)\s+",
                line,
            )

            if not match:
                continue

            found = match.group(1)
            found_parts = found.split(".")

            # Stop at 7.4.2, 7.5, 7.6, etc.
            if len(found_parts) == 2:
                end_index = i
                break

            # Stop at another sibling, e.g. 7.4.2
            if (
                len(found_parts) == 3
                and found_parts[1] == section
                and found_parts[2] != subsection
            ):
                end_index = i
                break

        # If this is 7.4.2, 7.5 etc. the above handles siblings.
    
    return {
        "target": target,
        "start": start_index + 1,
        "end": end_index,
        "lines": lines[start_index:end_index],
    }


def main():
    print("=" * 75)
    print("3GPP NORMATIVE RECEIVER REQUIREMENT INSPECTION")
    print("=" * 75)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE}"
        )

    lines = SOURCE.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    results = []

    for target in TARGETS:
        print()
        print("-" * 75)
        print(f"Clause: {target}")

        result = extract_clause(lines, target)

        if result is None:
            print("STATUS: NOT FOUND")
            continue

        results.append(result)

        content = result["lines"]

        print("STATUS: FOUND")
        print(
            f"Source lines: "
            f"{result['start']} -> {result['end']}"
        )
        print(f"Extracted lines: {len(content)}")
        print()

        for line in content[:8]:
            print(f"  {line}")

        if len(content) > 8:
            print("  ...")

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 75 + "\n")
        f.write("3GPP NORMATIVE RECEIVER REQUIREMENT REVIEW\n")
        f.write("=" * 75 + "\n\n")

        for result in results:
            f.write("-" * 75 + "\n")
            f.write(f"CLAUSE: {result['target']}\n")
            f.write(
                f"START SOURCE LINE: {result['start']}\n"
            )
            f.write(
                f"END SOURCE LINE: {result['end']}\n"
            )
            f.write(
                f"EXTRACTED LINES: {len(result['lines'])}\n"
            )
            f.write("-" * 75 + "\n\n")

            for line in result["lines"]:
                f.write(line + "\n")

            f.write("\n")

    print()
    print("=" * 75)
    print("VALIDATION COMPLETE")
    print("=" * 75)
    print(f"Targets:   {len(TARGETS)}")
    print(f"Extracted: {len(results)}")
    print(f"Failed:    {len(TARGETS) - len(results)}")
    print()
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()