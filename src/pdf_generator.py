import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(results, query, title="USAA Semantic Search Project"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"{title} – UNC Charlotte", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Semantic Search Results for: {query}", styles["Heading2"]))
    story.append(Spacer(1, 12))

    for i, r in enumerate(results, 1):
        title = r.get("title", "(No Title)")
        score = round(r.get("similarity", 0.0), 3)
        url = r.get("url", "N/A")
        date = r.get("date", "N/A")
        content = (r.get("content") or "")[:600] + "..."
        story.append(Paragraph(f"{i}. {title}", styles["Heading3"]))
        story.append(Paragraph(f"Score: {score} | Date: {date}", styles["Normal"]))
        story.append(Paragraph(f"URL: <a href='{url}'>{url}</a>", styles["Normal"]))
        story.append(Paragraph(content, styles["BodyText"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer
