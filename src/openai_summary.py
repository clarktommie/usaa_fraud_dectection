from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str):
    """
    Use OpenAI to produce a cohesive, narrative-style summary paragraph
    with inline citations based on referenced material in the input text.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an analytical writer specializing in financial compliance and fraud analysis. "
                        "Write 2–4 paragraphs summarizing the provided content in a cohesive narrative style. "
                        "If the text references or quotes external sources, include brief inline citations "
                        "(e.g., 'according to Smith (2024)' or '(Bloomberg, 2023)'). "
                        "Do not fabricate sources—only cite ones explicitly mentioned in the text. "
                        "Use clear, publication-ready prose."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.6,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {e}"

