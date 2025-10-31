"""
openai_summary.py
-----------------
Summarizes text or article content using the OpenAI API.
Usage:
    uv run python src/openai_summary.py "Your article text here"
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file so uv run sees your API key
load_dotenv()

# --- Setup ---
# make sure you have your API key set in the environment first:
#   export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_text(text: str, model: str = "gpt-4o-mini") -> str:
    """
    Summarize long text using OpenAI's GPT models.
    Defaults to gpt-4o-mini for low-cost, high-speed summarization.
    """
    if not text or len(text.strip()) == 0:
        return "No text provided for summarization."

    # model will automatically handle long text up to its token limit
    prompt = f"Summarize the following text in clear, concise form:\n\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


# --- Command-line testing ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/openai_summary.py \"Your article text here\"")
        sys.exit(1)

    input_text = sys.argv[1]
    summary = summarize_text(input_text)
    print("\n=== SUMMARY ===\n")
    print(summary)
