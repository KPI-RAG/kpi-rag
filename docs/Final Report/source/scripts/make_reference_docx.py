"""Create reference.docx (styles for pandoc): A4, 2.5 cm margins, serif body, page-number footer."""
import os
import subprocess

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reference.docx")
NAVY = RGBColor(0x1F, 0x38, 0x64)

with open(OUT, "wb") as f:
    f.write(subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"],
                           capture_output=True, check=True).stdout)

doc = Document(OUT)
for section in doc.sections:
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(2.5))
    footer = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    run.font.size = Pt(9)


def style(name, font="Cambria", size=None, bold=None, italic=None, color=None, before=None, after=None,
          align=None, line=None):
    s = doc.styles[name]
    s.font.name = font
    s.element.rPr.rFonts.set(qn("w:eastAsia"), font)
    s.element.rPr.rFonts.set(qn("w:ascii"), font)
    s.element.rPr.rFonts.set(qn("w:hAnsi"), font)
    if size: s.font.size = Pt(size)
    if bold is not None: s.font.bold = bold
    if italic is not None: s.font.italic = italic
    if color is not None: s.font.color.rgb = color
    pf = s.paragraph_format
    if before is not None: pf.space_before = Pt(before)
    if after is not None: pf.space_after = Pt(after)
    if align is not None: pf.alignment = align
    if line is not None: pf.line_spacing = line


style("Normal", size=11, line=1.15, after=6)
for name in ("Body Text", "First Paragraph", "Compact"):
    if name in [s.name for s in doc.styles]:
        style(name, size=11, line=1.15, after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY if name != "Compact" else None)
style("Title", size=22, bold=True, color=NAVY, before=24, after=12)
style("Heading 1", size=20, bold=True, color=NAVY, before=18, after=10)
style("Heading 2", size=16, bold=True, color=NAVY, before=18, after=8)
style("Heading 3", size=12.5, bold=True, color=NAVY, before=12, after=4)
style("Heading 4", size=11, bold=True, italic=False, color=NAVY, before=10, after=2)
for name in ("Image Caption", "Table Caption", "Caption"):
    if name in [s.name for s in doc.styles]:
        style(name, size=9.5, italic=True, color=RGBColor(0x40, 0x40, 0x48), before=4, after=10)
# table cells use "Compact": left-aligned and smaller than body text
style("Compact", size=9.5, line=1.0, after=2, align=WD_ALIGN_PARAGRAPH.LEFT)


def table_rules(style_id: str) -> None:
    """Booktabs-like rules (top, header, bottom) and a bold, shaded header row for pandoc's table style."""
    st = next(s for s in doc.styles.element.findall(qn("w:style")) if s.get(qn("w:styleId")) == style_id)
    tblPr = st.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        st.append(tblPr)
    borders = OxmlElement("w:tblBorders")
    for edge, size in (("top", 8), ("bottom", 8), ("insideH", 2)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), "8E8E99" if edge == "insideH" else "1C1C1E")
        borders.append(el)
    tblPr.append(borders)
    first = OxmlElement("w:tblStylePr")
    first.set(qn("w:type"), "firstRow")
    rpr = OxmlElement("w:rPr")
    rpr.append(OxmlElement("w:b"))
    first.append(rpr)
    tcpr = OxmlElement("w:tcPr")
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "DCE6F7")
    tcpr.append(shd)
    first.append(tcpr)
    st.append(first)


table_rules("Table")
doc.save(OUT)
print("wrote", OUT)
