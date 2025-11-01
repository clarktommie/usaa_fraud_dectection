from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str):
    """
    Use OpenAI to produce a *narrative-style* summary paragraph,
    not a numbered or bullet list.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an analytical writer who summarizes fraud and compliance patterns "
                        "in clear, cohesive narrative paragraphs. Avoid lists or bullet points, unless needed"
                        "Blend key observations naturally into a flowing explanation of trends and context."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {e}"
