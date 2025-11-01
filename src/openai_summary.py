from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str):
    """
    Use OpenAI to produce a cohesive, narrative-style summary paragraph.
    Avoid lists or bullet points — focus on analytical flow and full context.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o for higher accuracy if preferred
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an analytical writer specializing in financial compliance and fraud analysis. "
                        "Write 2–4 full paragraphs that summarize the provided content into a professional narrative. "
                        "Avoid bullet points, lists, or headings. Blend insights into continuous prose using "
                        "precise, publication-ready language suitable for a quarterly regulatory report."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.6,
            max_tokens=1200,  # Increased to prevent mid-summary cutoffs
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {e}"
