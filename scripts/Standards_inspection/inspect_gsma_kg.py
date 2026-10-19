import json
from pathlib import Path

path = Path(
    r"data\3GPP_KG\chunks\rel19_text_chunks.jsonl"
)

total = 0
non_empty = 0
empty = 0

series = {}
ts_counts = {}

with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        total += 1

        text = record.get("text", "")

        if text and text.strip():
            non_empty += 1
        else:
            empty += 1

        s = record.get("series", "UNKNOWN")
        series[s] = series.get(s, 0) + 1

        ts = record.get("ts", "UNKNOWN")
        ts_counts[ts] = ts_counts.get(ts, 0) + 1

print("Total chunks:", total)
print("Non-empty text:", non_empty)
print("Empty text:", empty)

print("\nSeries:")
for key, value in sorted(series.items()):
    print(key, value)

print("\nNumber of unique TS identifiers:", len(ts_counts))