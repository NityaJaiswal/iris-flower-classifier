
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
    Table, TableStyle, HRFlowable, Image, Flowable, Preformatted
)
from xml.sax.saxutils import escape
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "documentation" / "project_report.md"
REPORTS_DIR = ROOT / "outputs" / "reports"

DOCX_PATH = REPORTS_DIR / "Iris_Classification_Project_Report.docx"
PDF_PATH = REPORTS_DIR / "Iris_Classification_Project_Report.pdf"

FIGURES_DIR = ROOT / "documentation" / "report_figures"

FIGURE_CATALOG = [
    ("histogram_sepal_length_cm.png", "Distribution of Sepal Length"),
    ("histogram_sepal_width_cm.png", "Distribution of Sepal Width"),
    ("histogram_petal_length_cm.png", "Distribution of Petal Length"),
    ("histogram_petal_width_cm.png", "Distribution of Petal Width"),
    ("boxplot_sepal_length_cm.png", "Boxplot of Sepal Length"),
    ("boxplot_sepal_width_cm.png", "Boxplot of Sepal Width"),
    ("boxplot_petal_length_cm.png", "Boxplot of Petal Length"),
    ("boxplot_petal_width_cm.png", "Boxplot of Petal Width"),
    ("pairplot.png", "Pairwise Feature Relationships"),
    ("correlation_heatmap.png", "Feature Correlation Heatmap"),
    ("species_distribution.png", "Iris Species Distribution"),
    ("ovr_confusion_matrix.png", "OvR Confusion Matrix"),
    ("softmax_confusion_matrix.png", "Softmax Confusion Matrix"),
    ("multiclass_roc_curves.png", "Multiclass ROC Curves"),
    ("multiclass_precision_recall.png", "Multiclass Precision-Recall Curves"),
    ("decision_boundary_ovr_sepal_length_cm_sepal_width_cm.png", "OvR Decision Boundary: Sepal Features"),
    ("decision_boundary_softmax_sepal_length_cm_sepal_width_cm.png", "Softmax Decision Boundary: Sepal Features"),
    ("decision_boundary_ovr_petal_length_cm_petal_width_cm.png", "OvR Decision Boundary: Petal Features"),
    ("decision_boundary_softmax_petal_length_cm_petal_width_cm.png", "Softmax Decision Boundary: Petal Features"),
    ("ovr_vs_softmax_loss_convergence.png", "OvR vs Softmax Loss Convergence"),
    ("feature_coefficient_importance.png", "Mean Absolute Feature Coefficient Comparison"),
    ("model_coefficient_heatmaps.png", "Class-wise OvR and Softmax Coefficient Heatmaps"),
]

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

def add_docx_figure_appendix(doc):
    """Add all project figures to the Word report."""

    doc.add_page_break()
    doc.add_heading("APPENDIX D: PROJECT FIGURES", level=1)

    for index, (filename, caption) in enumerate(FIGURE_CATALOG, start=1):
        image_path = FIGURES_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"Required report figure not found: {image_path}"
            )

        # Keep each figure on its own page.
        if index > 1:
            doc.add_page_break()

        image_paragraph = doc.add_paragraph()
        image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # Fit both landscape and portrait figures within a readable page area.
        with PILImage.open(image_path) as source_image:
            width_px, height_px = source_image.size
        scale = min(5.9 / width_px, 6.6 / height_px)
        display_width = Inches(width_px * scale)
        display_height = Inches(height_px * scale)
        image_paragraph.add_run().add_picture(
            str(image_path), width=display_width, height=display_height
        )

        caption_paragraph = doc.add_paragraph()
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        caption_run = caption_paragraph.add_run(
            f"Figure D.{index}: {caption}"
        )
        caption_run.bold = True
        caption_run.font.size = Pt(10)

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

        # Start the Table of Contents on a new page.
    doc.add_page_break()

    # Static Table of Contents: no Word field update required.
    toc_heading = doc.add_heading("TABLE OF CONTENTS", level=1)
    toc_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    toc_entries = [
        ("DECLARATION", 0),
        ("ACKNOWLEDGEMENT", 0),
        ("ABSTRACT", 0),
        ("CHAPTER 1: INTRODUCTION", 0),
        ("1.1 Background", 1),
        ("1.2 Problem Statement", 1),
        ("1.3 Objectives", 1),
        ("1.4 Scope", 1),
        ("1.5 Software and Technologies Used", 1),
        ("CHAPTER 2: DATASET DESCRIPTION AND EXPLORATORY DATA ANALYSIS", 0),
        ("2.1 Dataset Overview", 1),
        ("2.2 Input Features", 1),
        ("2.3 Target Classes", 1),
        ("2.4 Dataset Preparation", 1),
        ("2.5 Exploratory Data Analysis", 1),
        ("2.6 Statistical Observations", 1),
        ("2.7 Data Quality Findings", 1),
        ("2.8 EDA Visualizations", 1),
        ("CHAPTER 3: DATA PREPROCESSING", 0),
        ("3.1 Feature and Target Separation", 1),
        ("3.2 Train-Test Split", 1),
        ("3.3 Feature Standardization", 1),
        ("3.4 Preprocessing Configuration", 1),
        ("CHAPTER 4: METHODOLOGY AND ALGORITHM DESIGN", 0),
        ("4.1 Logistic Regression", 1),
        ("4.2 One-vs-Rest Classification", 1),
        ("4.3 Softmax Logistic Regression", 1),
        ("4.4 Difference Between OvR and Softmax", 1),
        ("4.5 Machine Learning Pipelines", 1),
        ("CHAPTER 5: IMPLEMENTATION AND HYPERPARAMETER OPTIMIZATION", 0),
        ("5.1 Development Environment", 1),
        ("5.2 Model Training", 1),
        ("5.3 Hyperparameter Tuning", 1),
        ("5.4 Hyperparameter Optimization Results", 1),
        ("5.5 Model Serialization", 1),
        ("5.6 Loss Convergence Comparison", 1),
        ("CHAPTER 6: RESULTS AND PERFORMANCE EVALUATION", 0),
        ("6.1 Evaluation Method", 1),
        ("6.2 Evaluation Metrics", 1),
        ("6.3 Test Set Results", 1),
        ("6.4 Class-wise Classification Results", 1),
        ("6.5 Confusion Matrix Analysis", 1),
        ("6.6 ROC and Precision-Recall Analysis", 1),
        ("6.7 Decision Boundary Visualization", 1),
        ("6.8 Comparative Analysis", 1),
        ("6.9 Feature Coefficient Analysis and Model Insights", 1),
        ("CHAPTER 7: INTERACTIVE PREDICTION SYSTEM", 0),
        ("7.1 System Overview", 1),
        ("7.2 User Input", 1),
        ("7.3 Prediction Process", 1),
        ("7.4 Sample Prediction", 1),
        ("CHAPTER 8: TESTING AND VALIDATION", 0),
        ("8.1 Testing Approach", 1),
        ("8.2 Automated Test Cases", 1),
        ("8.3 Test Execution Results", 1),
        ("CHAPTER 9: CONCLUSION AND FUTURE SCOPE", 0),
        ("9.1 Conclusion", 1),
        ("9.2 Limitations", 1),
        ("9.3 Future Scope", 1),
        ("CHAPTER 10: REFERENCES", 0),
        ("LIST OF FIGURES AND TABLES", 0),
        ("APPENDIX A: PROJECT DIRECTORY STRUCTURE", 0),
        ("APPENDIX B: IMPORTANT EXECUTION COMMANDS", 0),
        ("APPENDIX C: CORE IMPLEMENTATION CODE", 0),
        ("APPENDIX D: PROJECT FIGURES", 0),
    ]

    for entry, level in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25 if level else 0)
        p.paragraph_format.space_after = Pt(3)

        run = p.add_run(entry)
        run.font.name = "Times New Roman"
        run.font.size = Pt(9.5 if level else 10.5)
        run.bold = (level == 0)

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

        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            code_paragraph = doc.add_paragraph()
            code_paragraph.paragraph_format.left_indent = Inches(0.15)
            code_paragraph.paragraph_format.right_indent = Inches(0.15)
            code_paragraph.paragraph_format.space_before = Pt(4)
            code_paragraph.paragraph_format.space_after = Pt(8)
            for code_index, code_line in enumerate(code_lines):
                if code_index:
                    code_paragraph.add_run().add_break()
                code_run = code_paragraph.add_run(code_line or " ")
                code_run.font.name = "Consolas"
                code_run.font.size = Pt(8)
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

        elif line.startswith("- "):
            content = line[2:]
            p = doc.add_paragraph(style="List Bullet")
            add_word_runs(p, content)

        elif re.match(r"^\d+\.\s", line):
            # Preserve Markdown numbering and prevent Word from
            # continuing automatic numbering across separate lists.
            p = doc.add_paragraph()
            add_word_runs(p, line)
            p.paragraph_format.left_indent = Inches(0.25)

        else:
            p = doc.add_paragraph()
            add_word_runs(p, line)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        i += 1

    add_docx_figure_appendix(doc)
    
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

def add_pdf_figure_appendix(story, styles):
    """Add all project figures to the PDF report."""

    story.append(PageBreak())
    story.append(Paragraph(
        "APPENDIX D: PROJECT FIGURES",
        styles["Chapter"]
    ))
    story.append(Spacer(1, 15))

    for index, (filename, caption) in enumerate(
        FIGURE_CATALOG, start=1
    ):
        image_path = FIGURES_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"Required report figure not found: {image_path}"
            )

        if index > 1:
            story.append(PageBreak())

        # Scale each figure to the largest readable size within the page frame.
        from PIL import Image as PILImage
        with PILImage.open(image_path) as source_image:
            image_width, image_height = source_image.size
        max_width, max_height = 6.9 * inch, 7.1 * inch
        scale = min(max_width / image_width, max_height / image_height)
        figure = Image(
            str(image_path),
            width=image_width * scale,
            height=image_height * scale
        )
        figure.hAlign = "CENTER"

        story.append(figure)
        story.append(Spacer(1, 12))

        story.append(Paragraph(
            f"<b>Figure D.{index}: {escape(caption)}</b>",
            styles["SmallReport"]
        ))

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

    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=9,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=5,
        spaceAfter=9,
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

        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            story.append(Preformatted("\n".join(code_lines), styles["CodeBlock"]))
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
            numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
            if numbered:
                marker, content = f"{numbered.group(1)}.", numbered.group(2)
            else:
                marker, content = "&#8226;", line[2:]
            story.append(Paragraph(
                marker + " " + pdf_inline(content),
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

    add_pdf_figure_appendix(story, styles)
    
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
