# Dockerfile — CPU-only image for the Arabic character recognizer demo.
#
# CPU-only by design: inference on a single 32x32 image takes milliseconds on
# CPU. The GPU is only needed for TRAINING, which happens outside the
# container. A CUDA image would add ~4GB for no benefit here, and would not
# run on most hosts.
#
# Build:  docker build -t arabic-ocr .
# Run:    docker run -p 8501:8501 arabic-ocr
# Then open http://localhost:8501

FROM python:3.11-slim

# Keep Python lean and unbuffered so logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed by Pillow / OpenCV-style image handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch wheel FIRST, from the CPU index.
# This is the key line: without --index-url you'd pull the CUDA build (~3GB).
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Then the rest of the app's dependencies.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Application code and the trained model.
COPY src/ ./src/
COPY app/ ./app/
COPY models/best_model.pt ./models/best_model.pt

EXPOSE 8501

# Streamlit needs these to serve correctly inside a container.
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Simple healthcheck against Streamlit's built-in endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py"]
