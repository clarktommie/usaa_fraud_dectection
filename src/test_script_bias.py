
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import fetch_articles

from src.data_loader import fetch_articles
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# ✅ Fetch live data from Supabase
all_articles = fetch_articles()

# Skip if nothing came back
if all_articles.empty:
    print("No articles returned from Supabase.")
else:
    # ✅ Count token frequencies
    cv = CountVectorizer(stop_words="english")
    X = cv.fit_transform(all_articles["content"].fillna(""))

    freqs = pd.DataFrame(X.sum(axis=0), columns=cv.get_feature_names_out())
    top_terms = freqs.T.sort_values(0, ascending=False).head(20)
    print(top_terms)

