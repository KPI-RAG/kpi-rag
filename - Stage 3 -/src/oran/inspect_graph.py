import ijson
from collections import Counter

FILE = r"data\oran\ORAN_Spec_Knowledge_graph.json"

label_pairs = Counter()
node_labels = Counter()
count = 0

print("Scanning O-RAN knowledge graph...")
print("This may take some time because the file is ~3.4 GB.\n")

with open(FILE, "rb") as f:
    for item in ijson.items(f, "item"):
        count += 1

        start = item.get("node_start", {})
        end = item.get("node_end", {})

        start_labels = tuple(start.get("labels", []))
        end_labels = tuple(end.get("labels", []))

        label_pairs[(start_labels, end_labels)] += 1

        for label in start_labels:
            node_labels[label] += 1

        for label in end_labels:
            node_labels[label] += 1

        if count % 10000 == 0:
            print(f"Processed {count:,} relationships...")

print("\n" + "=" * 60)
print(f"Total graph records: {count:,}")
print("=" * 60)

print("\nNode labels:")
for label, n in node_labels.most_common():
    print(f"  {label}: {n:,}")

print("\nNode-label relationships:")
for (start, end), n in label_pairs.most_common():
    print(f"  {start} -> {end}: {n:,}")