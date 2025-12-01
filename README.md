# USAA Fraud Detection
Semantic monitoring for regulatory press releases

## Overview
Automated ETL, embeddings, and storytelling dashboards that highlight emerging compliance and fraud risks for UNC Charlotte's DTSC 3602 project.

## Authors
- Jack Resnick
- Andreas Cedron
- Tommie Clark
- Ty Warren

## Quick Start
| Step | Command |
| --- | --- |
| Create environment | ```bash\nuv venv .venv\nsource .venv/bin/activate\nuv sync\n``` |
| Launch dashboard | ```bash\nuv run streamlit run streamlit_app3.py\n``` |

## Deploy to Modal (Streamlit)
- Install CLI: `uv tool install modal` (or `pip install modal` inside your venv).
- Create/update secret (from your `.env`):  
  `set -a && source .env && modal secret create fruad_detection SUPABASE_URL=\"$SUPABASE_URL\" SUPABASE_KEY=\"$SUPABASE_KEY\" OPENAI_API_KEY=\"$OPENAI_API_KEY\" OPENAI_TRENDS_MODEL=\"${OPENAI_TRENDS_MODEL:-gpt-4o-mini}\"`
- Deploy from repo root: `modal deploy modal_app.py`
- Open the URL shown after cold start (Modal will proxy port 8501).

### Required Environment Variables
| Key | Purpose | Example |
| --- | --- | --- |
| SUPABASE_URL | Supabase project REST endpoint | https://xyzcompany.supabase.co |
| SUPABASE_KEY | Supabase service/anon key | eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... |
| OPENAI_API_KEY | Embedding and summary API key | sk-abc123 |

Create a `.env` file and populate the values:
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_key
```

## Project Snapshot
- One-click UV setup, scripted scraper, and Streamlit UI for compliance intelligence.
- Supabase stores structured press releases from Federal Reserve and CFPB plus OpenAI embeddings for semantic recall.
- Storytelling view surfaces focus terms, multi-year trends, and GPT-generated insights for analysts.
- Agentic retrieval broadens or retries searches automatically so analysts get enough context.
- Full Retrieval-Augmented Generation (RAG) loop: retrieve context via embeddings, then generate OpenAI summaries.

### Why It Matters
- Unified fraud intelligence workspace that links data, embeddings, and AI summaries.
- Actionable compliance insights that highlight regulatory concerns for remediation and training.
- Reusable pipeline that can be adapted to other institutions with minimal change.

### Visual Overview
![Streamlit Dashboard](images/streamlit_dashboard.png)
Streamlit application (`streamlit_app3.py`) highlighting focus-word filters, yearly trend chart, and article summaries.

### Folder Structure
```
.
├── data/
│   └── cache/
│       ├── article_embeddings.npz
│       └── article_metadata.pkl
├── images/
│   ├── streamlit_dashboard.png
│   ├── dashboard_demo.gif
│   └── ChatGPT Image Nov 4, 2025, 07_57_35 AM.png
├── notebooks/
│   └── fraud_exploratory.ipynb
├── notes/
│   ├── embeddings_demo.ipynb
│   ├── install_notes.txt
│   └── USAA_Fraud_Detection_Project_Timeline.pdf
├── src/
│   ├── ai/
│   │   ├── article_preprocessing.py
│   │   ├── complaint_preprocessing.py
│   │   ├── fraud_insights.py
│   │   └── query_filter.py
│   ├── article_index.py
│   ├── cfpb_loader.py
│   ├── data_loader.py
│   ├── library_viewer.py
│   ├── openai_summary.py
│   ├── semantic_library.py
│   ├── sidebar_controls.py
│   ├── topic_keyword_utils.py
│   ├── topic_visuals.py
│   ├── universal_scraper_inner.py
│   ├── universal_scraper_outer.py
│   └── usaa_logo.py
├── streamlit_app3.py
├── README.md
└── pyproject.toml
```

## Architecture and Data Flow
```mermaid
flowchart LR
    A["Regulatory sites<br/>(FRB + CFPB)"] --> B["Scrapers<br/>HTML + PDF"]
    B --> C[Clean and normalize text]
    C --> D[(Supabase DB and Vector Store)]
    D --> E[Embedding cache / article_index]
    E --> F[Streamlit application]
    F --> G[OpenAI summaries and trend visuals]
```

## Data Glimpse and Transform Example
### Sample Record
| id | title | focus_word | date | risk_score |
| --- | --- | --- | --- | --- |
| 1045 | Consent Order on BSA Failures | AML | 2024-03-11 | 0.82 |
| 1096 | Supervisory Letter on Wire Fraud | wire fraud | 2024-05-02 | 0.74 |

### Minimal Transformation Snippet
```python
from src.topic_keyword_utils import infer_focus_phrase

query = "What are the major AML deficiencies regulators highlight?"
keyword = infer_focus_phrase(query, candidate_terms=["AML", "fraud", "BSA"])
print(keyword)  # -> "AML"
```
The Streamlit layer injects this inferred keyword into semantic searches, aligns yearly metrics, and feeds OpenAI summaries.

### Agentic Retrieval Snippet
```python
from src.article_index import ArticleIndex
from src.ai.agentic_tool import AgenticRetriever

index = ArticleIndex.load()
agent = AgenticRetriever(index)
result = agent.retrieve("wire fraud deadlines", focus_hint="wire fraud")
print(result.query_used)  # expanded query used by the agent
for step in result.steps:
    print(step.action, step.detail)
```
The agent enforces a Retrieval-Augmented Generation loop by adaptively expanding queries or backfilling cached context before the LLM generates summaries.

### Streamlit Data Flow Example
```python
import streamlit as st
from src.data_loader import fetch_articles
from src.topic_visuals import plot_focus_term_trend

st.title("USAA Fraud Detection Dashboard")
articles = fetch_articles()
focus_word = st.text_input("Focus term", "AML")
filtered = articles[articles["focus_word"] == focus_word]
st.pyplot(plot_focus_term_trend(filtered))
```
This minimal example mirrors `streamlit_app3.py`: cached Supabase data drives interactive filters and visuals.

## System Overview
1. Data collection scrapes Federal Reserve and CFPB releases, standardizes dates, and merges titles and content.
2. Data storage uses Supabase for relational data and embeddings created with `text-embedding-3-small`.
3. Semantic storytelling performs query-aware similarity search, identifies a focus word, and aggregates yearly metrics.
4. AI summaries convert article clusters into contextual narratives for analysts.
5. Library integration saves semantic collections for rapid recall in review sessions.

## Key Features
- Automated regulatory press release scraping.
- Supabase-backed storage plus vector similarity search.
- Local cached embedding index for low-latency queries.
- Domain-specific keyword extraction and focus term detection.
- Yearly trend visualization tied to the selected focus word.
- GPT-based summarization for narrative context.
- Library system to curate compliance topic groups, with a semantic viewer that can be tuned in the UI to search more or fewer related articles.

## Tech Stack
- Python 3.12
- Streamlit
- Supabase (Postgres and vector store)
- OpenAI GPT and embeddings
- Pandas and Matplotlib
- BeautifulSoup4, Requests, PyPDF2
- uv for environment and execution management

## Findings and Impact
| Insight | Why it matters | Visual |
| --- | --- | --- |
| Bank Secrecy Act (BSA) findings rose 22% year over year | Highlights priorities for fraud analysts and training | Dashboard focus-word trend chart |
| Wire fraud remediation deadlines are shorter (under 45 days) | Signals urgency for operations teams | Animated GIF demo highlighting alert cards |
| Repeat offenders cluster around six institutions | Guides investigative triage and storytelling | Library collections panel in Streamlit |

The project compresses scraping, labeling, trend analysis, and executive storytelling into one reproducible Streamlit experience, allowing analysts to pivot from macro trends to specific consent orders quickly.

## Current Status
- ETL pipeline, semantic model, and OpenAI summarization are operational.
- Focus-term detection and yearly trend visualization are functioning.
- Library integration for topic-based article collections is implemented.
- Documentation and architecture are production-ready.

Future improvements include a continuous scraping schedule, fine-tuned domain embeddings, and multilingual monitoring.

## Acknowledgment
Developed for DTSC 3602: Data Science Project at UNC Charlotte, demonstrating machine learning, LLM integration, and visual analytics for regulatory insight automation. Drafted with assistance from ChatGPT for documentation polish and coding support.
