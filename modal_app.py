"""
Modal deployment entry point for the Streamlit dashboard.

Usage:
- Create a Modal secret named `fruad_detection` containing SUPABASE_URL, SUPABASE_KEY,
  and OPENAI_API_KEY (optional OPENAI_TRENDS_MODEL).
- Deploy with `modal deploy modal_app.py` then open the provided https://<handle>.modal.run.
"""

from __future__ import annotations

import modal

app = modal.App("usaa-fraud-streamlit")

# Persist the article/embedding cache between warm restarts to avoid repeated rebuilds.
cache_volume = modal.Volume.from_name("usaa-fraud-cache", create_if_missing=True)

# Build a lightweight image with only the packages required to run the UI.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core")  # fonts for matplotlib/altair
    .pip_install(
        "altair>=5.5.0",
        "beautifulsoup4>=4.14.2",
        "matplotlib>=3.10.7",
        "numpy>=2.3.4",
        "openai>=2.6.1",
        "pandas>=2.3.3",
        "pydeck>=0.9.1",
        "python-dotenv>=1.1.1",
        "requests>=2.32.5",
        "scikit-learn>=1.7.2",
        "seaborn>=0.13.2",
        "streamlit>=1.50.0",
        "supabase>=2.22.1",
        "tqdm>=4.67.1",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=[
            ".git/**",
            ".venv/**",
            "__pycache__/**",
            "*.pyc",
            "data/cache/**",
            "node_modules/**",
        ],
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("fruad_detection")],
    volumes={"/root/app/data": cache_volume},
    timeout=1800,
    memory=8192,
    cpu=4,
)
@modal.web_server(port=8501, startup_timeout=60.0)
def serve():
    """Launch Streamlit inside Modal and expose it via the web server."""
    import os
    import subprocess

    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    # Streamlit will bind to port 8501 which is forwarded by the @web_server decorator.
    cmd = [
        "streamlit",
        "run",
        "streamlit_app3.py",
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
    ]
    # Start Streamlit and let the container keep running; Modal will route traffic to port 8501.
    subprocess.Popen(cmd, cwd="/root/app", env=env)


@app.local_entrypoint()
def main():
    """Run `modal run modal_app.py` to test locally via Modal before deploying."""
    serve.remote()
