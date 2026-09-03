# syntax=docker/dockerfile:1

# --- Stage 1: Build the React Frontend ---
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Build the FastAPI Backend ---
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by pymupdf / tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
# Step 1: Install CPU-only PyTorch (avoids downloading 1.5 GB of NVIDIA CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Step 2: Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- AI Model Caching ---
# Pre-download heavy AI models directly into the Docker image so they load instantly on Cloud Run
ENV FASTEMBED_CACHE_DIR=/opt/fastembed_cache
ENV HF_HOME=/opt/hf_cache
RUN python -c "import os; os.makedirs('/opt/fastembed_cache', exist_ok=True); os.makedirs('/opt/flashrank_cache', exist_ok=True); from fastembed import TextEmbedding, SparseTextEmbedding; from flashrank import Ranker; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed_cache'); SparseTextEmbedding('prithivida/Splade_PP_en_v1', cache_dir='/opt/fastembed_cache'); Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/opt/flashrank_cache')"

# Copy application source
COPY app/ ./app/

# Copy the built React frontend from Stage 1
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Cloud Run sets PORT env var at runtime — must be read dynamically
# Shell form (not exec form) is required to expand $PORT
ENV PORT=8080
EXPOSE $PORT

# Use shell form so ${PORT} is expanded from Cloud Run's runtime injection
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
