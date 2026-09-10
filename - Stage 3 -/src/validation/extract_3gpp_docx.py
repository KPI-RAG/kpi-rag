import sys
from pathlib import Path
from docx import Document

if len(sys.argv) < 2:
    print("Usage: python extract_3gpp_docx.py <docx_file>")
    sys.exit(1)

source = Path(sys.argv[1])
output = source.with_suffix(".txt")

print("=" * 70)
print("3GPP DOCUMENT EXTRACTION")
print("=" * 70)
print(f"Source: {source}")

doc = Document(source)

paragraphs = [p.text for p in doc.paragraphs]

text = "\n".join(paragraphs)

output.write_text(text, encoding="utf-8")

print(f"Paragraphs: {len(paragraphs)}")
print(f"Output: {output}")
print(f"Characters: {len(text):,}")