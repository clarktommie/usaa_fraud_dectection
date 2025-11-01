# src/modal_app.py
import modal
from pathlib import Path

# --- Paths ---
# Make sure you run `modal deploy src/modal_app.py` from the project root: fraud_detection/
root_dir = Path(__file__).resolve().parent.parent
project_dir = root_dir / "usaa_fraud_detection"
streamlit_app = project_dir / "streamlit_app3.py"   # <-- fixed path
env_file = project_dir / ".env"                     # <-- fixed path

# --- Modal image configuration ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .uv_pip_install(
        "streamlit",
        "pandas",
        "openai",
        "supabase",
        "python-dotenv",
        "altair",
        "scikit-learn",
        "wordcloud",
    )
    # Copy your source files and app into container
    .add_local_dir(project_dir / "src", "/root/src")
    .add_local_file(streamlit_app, "/root/streamlit_app3.py")
    .add_local_file(env_file, "/root/.env")
)

# --- Modal app ---
app = modal.App(name="usaa-fraud-dashboard", image=image)

@app.function()
@modal.web_server(port=8501)
def serve():
    import subprocess
    subprocess.run([
        "streamlit", "run", "/root/streamlit_app3.py",
        "--server.port=8501", "--server.address=0.0.0.0"
    ])
