"""Assemble the merged final report and build MD, HTML and DOCX.

Inputs (in source/): parts listed in PARTS, REFERENCES.md (key -> IEEE entry).
Drafts use [@key] citations, ![Caption.](figures/x.png){#fig:id} figures, `Table: Caption. {#tab:id}`
table captions, and {fig:id} / {tab:id} cross-references. This script numbers all of them in order of
appearance, writes the reference list, the lists of figures and tables, and the three outputs.

usage: python build.py            (needs pypandoc_binary)
"""
import os
import re
import sys

import pypandoc

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WORK)
NAME = "KPI_RAG_Final_Report"
PARTS = ["front.md", "abbrev.md", "drafts/ch01_02.md", "drafts/ch03.md", "drafts/ch04.md", "drafts/ch05a.md",
         "drafts/ch05b.md", "drafts/ch06.md", "drafts/ch07.md", "@references", "appendices.md", "appendix_contrib.md"]
TOC_MARK = "<!-- TOC -->"
PAGEBREAK = "<!-- PAGEBREAK -->"

FIG_RE = re.compile(r"!\[(.+?)\]\((figures/[^)\s]+)\)\{#(fig:[\w-]+)\}")
TAB_RE = re.compile(r"^Table: (.+?) \{#(tab:[\w-]+)\}\s*$", re.M)
CITE_RE = re.compile(r"\[((?:@[\w-]+)(?:\s*[;,]\s*@[\w-]+)*)\]")
REF_RE = re.compile(r"\{((?:fig|tab):[\w-]+)\}")


def read(rel: str) -> str:
    with open(os.path.join(WORK, rel), encoding="utf-8") as f:
        return f.read().strip() + "\n"


def load_short_titles() -> dict[str, str]:
    path = os.path.join(WORK, "SHORT_TITLES.md")
    if not os.path.exists(path):
        return {}
    pairs = (l.split("|", 1) for l in read("SHORT_TITLES.md").splitlines() if re.match(r"^(fig|tab):", l))
    return {k.strip(): v.strip() for k, v in pairs}


SHORT_TITLES = load_short_titles()


def load_references() -> dict[str, str]:
    refs = {}
    for line in read("REFERENCES.md").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"[\w-]+", cells[0]) and cells[0] != "key":
            refs[cells[0]] = cells[1]
    return refs


def number(text: str, refs: dict[str, str]):
    """Number figures, tables and citations; return the new text and the three registries."""
    figs, tabs, cites, missing = {}, {}, {}, set()

    def fig(m):
        n = figs.setdefault(m.group(3), (len(figs) + 1, m.group(1)))[0]
        return f"![Figure {n}. {m.group(1)}]({m.group(2)}){{#{m.group(3).replace(':', '-')}}}"

    def tab(m):
        n = tabs.setdefault(m.group(2), (len(tabs) + 1, m.group(1)))[0]
        return f"Table: Table {n}. {m.group(1)} {{#{m.group(2).replace(':', '-')}}}"

    def cite(m):
        nums = []
        for key in re.findall(r"@([\w-]+)", m.group(1)):
            if key not in refs:
                missing.add(key)
                continue
            nums.append(cites.setdefault(key, len(cites) + 1))
        return "[" + ", ".join(str(n) for n in sorted(set(nums))) + "]" if nums else ""

    text = FIG_RE.sub(fig, text)
    text = TAB_RE.sub(tab, text)
    text = CITE_RE.sub(cite, text)

    def xref(m):
        reg = figs if m.group(1).startswith("fig") else tabs
        if m.group(1) not in reg:
            missing.add(m.group(1))
            return m.group(0)
        word = "Figure" if m.group(1).startswith("fig") else "Table"
        return f"{word} {reg[m.group(1)][0]}"

    text = REF_RE.sub(xref, text)
    return text, figs, tabs, cites, missing


def headings(text: str):
    """(level, title, gfm slug) for ## and ### headings outside code blocks."""
    out, in_code = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{2,3}) (.+?)\s*$", line)
        if m:
            title = m.group(2)
            slug = re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")
            out.append((len(m.group(1)), title, slug))
    return out


def short(caption: str, limit: int = 16) -> str:
    """Short list-of-figures/tables title: the caption's first sentence, cut at a clause break if long."""
    first = re.split(r"(?<=[.!?])\s", caption.strip(), maxsplit=1)[0].rstrip(".")
    words = first.split()
    if len(words) <= limit:
        return first
    for i, w in enumerate(words[5:limit + 6], start=5):
        if w.endswith((",", ";", ":")):
            return " ".join(words[:i + 1]).rstrip(",;:")
    return " ".join(words[:limit])


def list_title(cid: str, caption: str) -> str:
    """Short title from SHORT_TITLES.md, else the automatic short form of the caption."""
    return SHORT_TITLES.get(cid) or short(caption)


def toc_block(text, figs, tabs) -> str:
    lines = ["## Table of Contents", ""]
    for level, title, slug in headings(text):
        if title == "Table of Contents":
            continue
        lines.append(("  " if level == 3 else "") + f"- [{title}](#{slug})")
    lines += ["", "### List of Figures", ""]
    lines += [f"- Figure {n}. {list_title(cid, cap)}" for cid, (n, cap) in sorted(figs.items(), key=lambda kv: kv[1][0])]
    lines += ["", "### List of Tables", ""]
    lines += [f"- Table {n}. {list_title(cid, cap)}" for cid, (n, cap) in sorted(tabs.items(), key=lambda kv: kv[1][0])]
    return "\n".join(lines) + "\n"


def lists_only(figs, tabs) -> str:
    lines = ["### List of Figures", ""] + [f"- Figure {n}. {list_title(cid, cap)}" for cid, (n, cap) in sorted(figs.items(), key=lambda kv: kv[1][0])]
    lines += ["", "### List of Tables", ""] + [f"- Table {n}. {list_title(cid, cap)}" for cid, (n, cap) in sorted(tabs.items(), key=lambda kv: kv[1][0])]
    return "\n".join(lines) + "\n"


def assemble():
    refs = load_references()
    parts = []
    for p in PARTS:
        parts.append("@REFS@\n" if p == "@references" else read(p))
    text = "\n\n".join(parts)
    text, figs, tabs, cites, missing = number(text, refs)
    reflist = "## References\n\n" + "\n\n".join(
        f"[{n}] {refs[k]}" for k, n in sorted(cites.items(), key=lambda kv: kv[1])) + "\n"
    text = text.replace("@REFS@\n", reflist)
    return text, figs, tabs, cites, missing


DOCX_TOC = """```{=openxml}
<w:p><w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr><w:r><w:t>Table of Contents</w:t></w:r></w:p>
<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "2-3" \\h \\z \\u</w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Right-click and choose Update Field to show page numbers.</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
```
"""
FONTS_LINK = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@'
              '0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,400&amp;family=Inter:wght@400;500;600&amp;display=swap">')
DOCX_BREAK ='```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'


def main() -> int:
    text, figs, tabs, cites, missing = assemble()
    if missing:
        print("UNRESOLVED:", sorted(missing))

    # Markdown (GitHub-readable): linked contents list, no pandoc attributes
    md = text.replace(TOC_MARK, toc_block(text, figs, tabs)).replace(PAGEBREAK + "\n", "")
    md = re.sub(r"(\]\(figures/[^)]+\))\{#[\w-]+\}", r"\1", md)
    md = re.sub(r" \{#tab-[\w-]+\}$", "", md, flags=re.M)
    # GitHub shows neither pandoc table captions nor image alt text: write both as visible caption lines
    md = re.sub(r"^Table: (Table \d+\.) (.+)$", r"**\1** \2\n", md, flags=re.M)
    md = re.sub(r"^!\[(Figure \d+\.) (.+?)\]\((figures/[^)]+)\)$", r"![\1](\3)\n\n**\1** \2", md, flags=re.M)
    md = re.sub(r"^:::.*\n", "", md, flags=re.M)                     # fenced divs (title page)
    md = re.sub(r"\[([^\]]+)\]\{\.[\w-]+\}", r"*\1*", md)            # [text]{.class} spans
    with open(os.path.join(ROOT, NAME + ".md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    # HTML: same contents list, report styling, figures embedded
    html_md = text.replace(TOC_MARK, toc_block(text, figs, tabs)).replace(
        PAGEBREAK, '<div class="pagebreak"></div>')
    pypandoc.convert_text(
        html_md, "html5", format="markdown+gfm_auto_identifiers+pipe_tables+tex_math_dollars",
        outputfile=os.path.join(ROOT, NAME + ".html"),
        extra_args=["--standalone", "--embed-resources", f"--resource-path={ROOT}",
                    f"--css={os.path.join(WORK, 'scripts', 'report.css')}", "--mathml",
                    "--metadata", "pagetitle=KPI-RAG Final Report",
                    "--metadata", "lang=en-GB"])

    # Fonts are linked, not embedded (embedding would fetch them at build time); offline readers get Georgia.
    html_path = os.path.join(ROOT, NAME + ".html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</head>", FONTS_LINK + "\n</head>", 1)
    with open(html_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)

    # DOCX: Word TOC field, page breaks, figures at text width
    lists = lists_only(figs, tabs).replace("### List of", "## List of")   # top-level entries in Word's TOC
    # one paragraph per entry, no bullets (a one-line paragraph is not stretched by justification)
    lists = re.sub(r"^- ((?:Figure|Table) \d+\..*)$", r"\1\n", lists, flags=re.M)
    docx_md = text.replace(TOC_MARK, DOCX_TOC + "\n" + lists).replace(PAGEBREAK, DOCX_BREAK)
    # the report title is a Title paragraph in Word, not a heading (keeps it out of the TOC)
    docx_md = re.sub(r"^# (.+)$", r'::: {custom-style="Title"}\n\1\n:::', docx_md, count=1, flags=re.M)
    docx_md = re.sub(r"(\]\(figures/[^)]+\))\{#([\w-]+)\}", r"\1{#\2 width=100%}", docx_md)
    ref_doc = os.path.join(WORK, "scripts", "reference.docx")
    args = [f"--resource-path={ROOT}"] + ([f"--reference-doc={ref_doc}"] if os.path.exists(ref_doc) else [])
    pypandoc.convert_text(docx_md, "docx", format="markdown+gfm_auto_identifiers+pipe_tables+tex_math_dollars",
                          outputfile=os.path.join(ROOT, NAME + ".docx"), extra_args=args)
    print(f"built: {len(figs)} figures, {len(tabs)} tables, {len(cites)} references, "
          f"{len(md.split())} words")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
