# USAA Fraud Detection

## Team Members
- Jack Resnick  
- Andreas Cedron  
- Tommie Clark  
- Ty Warren  

**DTSC 3602 | UNC Charlotte**

---

## Project Summary
This repository contains the completed **USAA Fraud Detection and Storytelling Dashboard**, developed as part of **DTSC 3602 (Data Science Project)** at UNC Charlotte.  

The project automates the collection, analysis, and semantic interpretation of **financial compliance and fraud-related press releases** from the **Federal Reserve Board** and related regulators.  
It integrates **embedding-based semantic search** with **OpenAI-powered summarization** to identify compliance themes, trends, and emerging risks across multiple years of financial reports.

---

## System Overview

### 🧠 Core Workflow
1. **Data Collection**
   - Scrapes official press releases from the Federal Reserve Board website.  
   - Extracts text from both HTML and PDF files using BeautifulSoup and PyPDF.  
   - Cleans and stores structured results in **Supabase** (`press_releases_clean` table).  

2. **Data Storage**
   - Supabase acts as both the **relational database** and **vector store**.  
   - Articles are embedded using **SentenceTransformer** (`all-MiniLM-L6-v2`) and stored as high-dimensional vectors for similarity search.

3. **Semantic Storytelling**
   - A **semantic model** ranks articles by relevance to a user query.  
   - The system detects a **focus word** (e.g., “AML”, “phishing”, “compliance”) using a domain dictionary of fraud and regulatory terms.  
   - Patterns and yearly trends are visualized through Streamlit.

4. **AI-Powered Summarization**
   - The **OpenAI API** generates natural-language summaries highlighting key insights, regulatory tone, emerging risk areas, and sentiment context.  
   - Summaries are derived directly from the articles and yearly trend data.

5. **Library Integration**
   - Related article collections are stored as “semantic groups” in a Supabase table (`semantic_collections`) for later retrieval and review.  
   - A built-in **Library Viewer** lets users explore saved topic clusters interactively.

---

## Key Features
✅ Automated regulatory press release scraping  
✅ Supabase integration for centralized storage and embeddings  
✅ SentenceTransformer-based semantic retrieval  
✅ Domain-specific keyword and phrase extraction  
✅ Yearly trend visualization for any detected focus word  
✅ OpenAI-driven contextual storytelling summaries  
✅ Library system for curated article collections  

---

## Tech Stack
- **Python 3.12**  
- **Streamlit** – interactive dashboard interface  
- **Supabase** – data storage and vector similarity search  
- **SentenceTransformer (all-MiniLM-L6-v2)** – semantic embeddings  
- **OpenAI GPT Models** – summarization and focus-word reasoning  
- **Pandas**, **Matplotlib** – analytics and visualization  
- **BeautifulSoup4**, **Requests**, **PyPDF2** – data scraping and extraction  

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/usaa_fraud_detection.git
cd usaa_fraud_detection
```

### 2. Create Environment
Using **uv** (preferred) or Conda:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root with:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_key
```

### 4. Run the Streamlit App
```bash
uv run streamlit run streamlit_app3.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Current Status
✅ **Semantic model, embeddings, and article pipeline are fully operational**  
✅ **OpenAI summarization produces contextually accurate insights**  
✅ **Focus-term detection and yearly trend visualization functioning correctly**  
✅ **Library integration for topic-based article collections implemented**  

Future updates (e.g., real-time scraping, fine-tuned embeddings, multi-language support) will be logged in future revisions.

---

## Acknowledgment
Developed for **DTSC 3602: Data Science Project at UNC Charlotte**  
Demonstrates applied data science using **machine learning**, **LLM integration**, and **visual analytics** for regulatory insight automation.