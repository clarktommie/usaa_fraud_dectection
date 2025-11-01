# src/modal_app.py
import modal
from pathlib import Path
import subprocess

# --- Paths ---
root_dir = Path(__file__).resolve().parent.parent
project_dir = root_dir / "usaa_fraud_detection"
streamlit_app = project_dir / "streamlit_app3.py"
env_file = project_dir / ".env"

# --- Modal image configuration ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    # Dependencies matched from pyproject.toml
    .uv_pip_install(
        "altair>=5.5.0",
        "beautifulsoup4>=4.14.2",
        "dotenv>=0.9.9",
        "fastembed>=0.7.3",
        "feedparser>=6.0.12",
        "matplotlib>=3.10.7",
        "networkx>=3.5",
        "numpy>=2.3.4",
        "openai>=2.6.1",
        "pandas>=2.3.3",
        "pdfminer-six>=20250506",
        "plotly>=6.3.1",
        "pypdf2>=3.0.1",
        "python-dotenv>=1.1.1",
        "reportlab>=4.4.4",
        "requests>=2.32.5",
        "scikit-learn>=1.7.2",
        "seaborn>=0.13.2",
        "sentence-transformers==2.7.0",
        "streamlit>=1.50.0",
        "supabase>=2.22.1",
        "textblob>=0.19.0",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "tqdm>=4.67.1",
        "transformers==4.44.2",
        "wordcloud>=1.9.4",
    )
    .add_local_dir(project_dir / "src", "/root/src")
    .add_local_file(streamlit_app, "/root/streamlit_app3.py")
    .add_local_file(env_file, "/root/.env")
)

# --- Modal app ---
app = modal.App(name="usaa-fraud-dashboard", image=image)

@app.function(timeout=600, startup_timeout=600)
@modal.web_server(port=8000)
def serve():
    # Run Streamlit non-blocking so Modal doesn't time out
    subprocess.Popen(
        [
            "streamlit", "run", "/root/streamlit_app3.py",
            "--server.port=8000", "--server.address=0.0.0.0"
        ]
    )
    print("✅ Streamlit server started on port 8000")
