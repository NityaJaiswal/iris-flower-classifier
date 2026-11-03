
from pathlib import Path
import re

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable
)
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "documentation" / "project_report.md"
REPORTS_DIR = ROOT / "outputs" / "reports"

DOCX_PATH = REPORTS_DIR / "Iris_Classification_Project_Report.docx"
PDF_PATH = REPORTS_DIR / "Iris_Classification_Project_Report.pdf"


def read_markdown():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Report source not found: {SOURCE}")
    return SOURCE.read_text(encoding="utf-8").splitlines()


def parse_inline(text):
    """Convert basic Markdown inline formatting into Word runs."""
    pattern = r"(\*\*.*?\*\*|`.*?`|\*.*?\*)"
    parts = re.split(pattern, text)

    result = []
    for part in parts:
        if not part:
            continue

        if part.startswith("**") and part.endswith("**"):
            result.append((part[2:-2], True, False))
        elif part.startswith("`") and part.endswith("`"):
            result.append((part[1:-1], False, True))
        elif part.startswith("*") and part.endswith("*"):
            result.append((part[1:-1], False, False))
        else:
            result.append((part, False, False))

    return result


def add_word_runs(paragraph, text):
    for content, bold, code in parse_inline(text):
        run = paragraph.add_run(content)
        run.bold = bold
        if code:
            run.font.name = "Consolas"
            run.font.size = Pt(9)


def shade_cell(cell, fill):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def set_word_page_number(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Page ")

    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def build_docx(lines):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

    set_word_page_number(section)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].paragraph_format.space_after = Pt(6)

    for style_name, size in [
        ("Heading 1", 17),
        ("Heading 2", 14),
        ("Heading 3", 12),
    ]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(31, 78, 121)

    # Professional title page
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(90)

    run = title.add_run("MULTICLASS IRIS FLOWER\nCLASSIFICATION")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(23)
    run.font.color.rgb = RGBColor(31, 78, 121)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run(
        "\nA Comparative Study of One-vs-Rest and "
        "Softmax Logistic Regression"
    )
    r.font.size = Pt(15)

    for value in [
    "\n\nPROJECT REPORT",
    "\n\nSubmitted in partial fulfilment of the requirements for",
    "\nBachelor of Technology",
    "\nComputer Science and Engineering – Artificial Intelligence and Machine Learning",
    "\n\nSubmitted by: Nitya Jaiswal",
    "\nRoll Number: 2402841530051",
    "\n\nUnder the Guidance of: Mr. Gaurav Narain Singh",
    "\n\nUnited Institute of Technology, Naini, Prayagraj",
    "\nDr. A.P.J. Abdul Kalam Technical University (AKTU), Uttar Pradesh",
    "\nAcademic Session: 2026–2027",
]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(value)

    doc.add_page_break()

    # Skip the Markdown title page because a formatted title page was added.
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            start = i + 1
            break

    i = start
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == "---":
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                current = lines[i].strip()
                if not re.match(r"^\|[\s\-:|]+\|$", current):
                    cells = [x.strip() for x in current.strip("|").split("|")]
                    table_lines.append(cells)
                i += 1

            if table_lines:
                table = doc.add_table(
                    rows=1,
                    cols=len(table_lines[0])
                )
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.style = "Table Grid"

                for row_index, row_data in enumerate(table_lines):
                    cells = (
                        table.rows[0].cells
                        if row_index == 0
                        else table.add_row().cells
                    )

                    for col_index, value in enumerate(row_data):
                        cell = cells[col_index]
                        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                        p = cell.paragraphs[0]
                        add_word_runs(p, value)

                        if row_index == 0:
                            shade_cell(cell, "D9EAF7")
                            for run in p.runs:
                                run.bold = True

                doc.add_paragraph()
            continue

        if line.startswith("# "):
            heading = line[2:].strip()

            if heading.startswith("CHAPTER "):
                doc.add_page_break()

            p = doc.add_heading(heading, level=1)
            p.paragraph_format.keep_with_next = True

        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)

        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)

        elif line.startswith("- ") or re.match(r"^\d+\.\s", line):
            if line.startswith("- "):
                content = line[2:]
                style = "List Bullet"
            else:
                content = re.sub(r"^\d+\.\s", "", line)
                style = "List Number"

            p = doc.add_paragraph(style=style)
            add_word_runs(p, content)

        else:
            p = doc.add_paragraph()
            add_word_runs(p, line)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        i += 1

    doc.save(DOCX_PATH)
    print(f"DOCX created: {DOCX_PATH}")


def pdf_inline(text):
    """Escape text and preserve simple Markdown emphasis."""
    text = escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.*?)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    return text


def add_pdf_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(A4[0] / 2, 0.4 * inch, str(document.page))
    canvas.restoreState()


def build_pdf(lines):
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=22,
        leading=29,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1F4E79"),
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name="Chapter",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=17,
        leading=22,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=10,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="Subheading",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=8,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="SmallReport",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
    ))

    story = []

    # PDF title page
    story.append(Spacer(1, 1.25 * inch))
    story.append(Paragraph(
        "MULTICLASS IRIS FLOWER<br/>CLASSIFICATION",
        styles["ReportTitle"]
    ))
    story.append(Paragraph(
        "A Comparative Study of One-vs-Rest and "
        "Softmax Logistic Regression",
        styles["ReportSubtitle"]
    ))
    story.append(Spacer(1, 0.5 * inch))

    title_lines = [
    "PROJECT REPORT",
    "Submitted in partial fulfilment of the requirements for",
    "Bachelor of Technology",
    "Computer Science and Engineering – Artificial Intelligence and Machine Learning",
    "Submitted by: Nitya Jaiswal",
    "Roll Number: 2402841530051",
    "Under the Guidance of: Mr. Gaurav Narain Singh",
    "United Institute of Technology, Naini, Prayagraj",
    "Dr. A.P.J. Abdul Kalam Technical University (AKTU), Uttar Pradesh",
    "Academic Session: 2026–2027",
]

    for value in title_lines:
        story.append(Paragraph(escape(value), styles["ReportSubtitle"]))
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    start = 0
    for index, line in enumerate(lines):
        if line.strip() == "---":
            start = index + 1
            break

    i = start
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == "---":
            i += 1
            continue

        if line.startswith("|"):
            table_data = []

            while i < len(lines) and lines[i].strip().startswith("|"):
                current = lines[i].strip()

                if not re.match(r"^\|[\s\-:|]+\|$", current):
                    cells = [
                        Paragraph(pdf_inline(x.strip()), styles["SmallReport"])
                        for x in current.strip("|").split("|")
                    ]
                    table_data.append(cells)

                i += 1

            if table_data:
                col_count = len(table_data[0])
                table = Table(
                    table_data,
                    colWidths=[6.8 * inch / col_count] * col_count,
                    repeatRows=1,
                    hAlign="CENTER"
                )
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(table)
                story.append(Spacer(1, 10))
            continue

        if line.startswith("# "):
            heading = line[2:].strip()

            if heading.startswith("CHAPTER "):
                story.append(PageBreak())

            story.append(Paragraph(
                pdf_inline(heading),
                styles["Chapter"]
            ))

        elif line.startswith("## "):
            story.append(Paragraph(
                pdf_inline(line[3:].strip()),
                styles["Subheading"]
            ))

        elif line.startswith("### "):
            story.append(Paragraph(
                pdf_inline(line[4:].strip()),
                styles["Heading3"]
            ))

        elif line.startswith("- ") or re.match(r"^\d+\.\s", line):
            content = re.sub(r"^(-|\d+\.)\s*", "", line)
            story.append(Paragraph(
                "&#8226; " + pdf_inline(content),
                styles["BodyReport"]
            ))

        else:
            story.append(Paragraph(
                pdf_inline(line),
                styles["BodyReport"]
            ))

        i += 1

    pdf = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title="Multiclass Iris Flower Classification Project Report",
        author="Nitya Jaiswal",
    )

    pdf.build(
        story,
        onFirstPage=add_pdf_page_number,
        onLaterPages=add_pdf_page_number
    )

    print(f"PDF created: {PDF_PATH}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = read_markdown()

    build_docx(lines)
    build_pdf(lines)

    print("\nBoth report files generated successfully!")


if __name__ == "__main__":
    main()
