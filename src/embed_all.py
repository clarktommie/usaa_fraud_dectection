import os
import time
import json
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment and initialize clients
# -------------------------------------------------
load_dotenv(".env")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# -------------------------------------------------
# Embedding helper for long text
# -------------------------------------------------
def embed_long_text(text, chunk_size=3000):
    """
    Splits long text into chunks so we never exceed token limits.
    Averages chunk embeddings to produce one vector.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    vectors = []

    for idx, chunk in enumerate(chunks):
        print(f"  Embedding chunk {idx + 1}/{len(chunks)} (size {len(chunk)} chars)")

        emb = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )

        vectors.append(emb.data[0].embedding)
        time.sleep(0.8)  # gentle rate-limit protection

    # Average vectors
    final_vector = [sum(vals) / len(vals) for vals in zip(*vectors)]
    return final_vector


# -------------------------------------------------
# Main embedding loop
# -------------------------------------------------
def process_embeddings(batch_size=25):
    """
    Pull rows with NULL embeddings and process them in batches.
    Resumes automatically where it left off.
    """

    while True:
        rows = (
            supabase.table("press_releases_clean")
            .select("id, content")
            .is_("embedding", "null")
            .limit(batch_size)
            .execute()
            .data
        )

        if not rows:
            print("\n✅ All rows are embedded. Finished!")
            break

        print(f"\nProcessing batch of {len(rows)} rows…")

        for row in rows:
            row_id = row["id"]
            text = row["content"] or ""

            print(f"\n[{row_id}] content length: {len(text)} chars")

            if len(text.strip()) == 0:
                print("  ⚠️ Empty content, storing empty embedding.")
                supabase.table("press_releases_clean").update(
                    {"embedding": []}
                ).eq("id", row_id).execute()
                continue

            try:
                if len(text) > 3000:
                    print("  Long article detected… chunking")
                    vector = embed_long_text(text)
                else:
                    print("  Short article… embedding")
                    emb = openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    vector = emb.data[0].embedding
                    print("TYPE CHECK:", type(vector), "length:", len(vector))

                # ✅ Store raw vector list (not JSON)
                supabase.table("press_releases_clean").update(
                    {"embedding": vector}
                ).eq("id", row_id).execute()

                print("  ✅ Saved embedding")

            except Exception as e:
                print(f"  ❌ Error embedding row {row_id}:", e)
                continue

            time.sleep(0.8)


# -------------------------------------------------
# Run the script
# -------------------------------------------------
if __name__ == "__main__":
    print("Starting embedding pipeline…")
    process_embeddings(batch_size=25)
