# Deploying Serverless ML: The Google Cloud Run Saga

Deploying a heavy, local Machine Learning pipeline to a serverless environment is notoriously difficult due to ephemeral storage, RAM limits, and cold-start timeouts. This document outlines the exhaustive DevOps debugging process required to successfully deploy the AskMyNotes application to Google Cloud Run.

## 1. Architectural Unification (The CORS Fix)
**The Problem:** The React frontend and FastAPI backend were running separately, causing persistent CORS pre-flight failures when communicating across domains in production.
**The Solution:** We transitioned to a **Unified Container Architecture**. 
- Rewrote the `Dockerfile` to use a Multi-Stage build.
- Stage 1 compiles the React Vite app into static assets.
- Stage 2 copies those static assets into the Python FastAPI container and mounts them using FastAPI's `StaticFiles`. 
- **Result:** Frontend and Backend now run on the exact same GCP URL, completely eliminating CORS.

## 2. Docker Context Issues
**The Problem:** The multi-stage build failed because Docker couldn't find the `frontend/` directory.
**The Fix:** We discovered the `.dockerignore` file was explicitly ignoring the `frontend` folder (a leftover from when they were separated). We updated `.dockerignore` to allow the source code but explicitly exclude `node_modules` to keep the build context small and fast.

## 3. Frontend Routing Mismatch
**The Problem:** In production, the React app was making requests to `/api/upload` resulting in `405 Method Not Allowed`.
**The Fix:** During local development, Vite proxies `/api` to the backend. In production, this proxy doesn't exist. We updated `App.jsx` to use `import.meta.env.PROD` to dynamically strip the `/api` prefix when running on Cloud Run.

## 4. The 500 Error: Hugging Face Downloads in Ephemeral Storage
**The Problem:** Hitting `/upload` returned a 500 Error: `Could not load model from any source.` 
**The Cause:** `fastembed` and `flashrank` attempt to download gigabytes of AI models from Hugging Face at runtime. Cloud Run mounts `/tmp` as a RAM disk, and the slow, heavy download during a web request caused timeouts and complete failures.
**The Fix (Baking Models):** We modified the `Dockerfile` to run a Python script during the GitHub Action build step. This pre-downloads all Dense, Sparse, and Re-ranking models directly into the Docker image at `/opt/`, ensuring they are instantly available at runtime without internet dependency.

## 5. The Catch-22: Library Default Caches
**The Problem:** Even after baking the models into the Docker image, the 500 error persisted.
**The Cause:** We discovered that `fastembed` ignores environment variables like `FASTEMBED_CACHE_DIR`. During the Docker build, it saved the models to the default `/root/` path. However, our runtime code explicitly looked for them in `/opt/`.
**The Fix:** We strictly passed `cache_dir='/opt/fastembed_cache'` into both the Dockerfile build script AND the runtime `app/ingest.py` code, ensuring the paths aligned perfectly.

## 6. The 503 Error: The OOM (Out of Memory) Crash
**The Problem:** With the models perfectly baked in, hitting `/upload` resulted in an instant 503 Service Unavailable.
**The Cause:** Because the models loaded instantly and successfully, the pipeline proceeded to parse the PDF. This caused the Dense Model, Sparse Model, Cross-Encoder, and PyMuPDF to all inhabit RAM simultaneously. This massive spike exceeded Cloud Run's default `512Mi` limit, causing Google Cloud to instantly OOM-Kill the container.
**The Fix:** We updated `.github/workflows/deploy.yml` to provision the Cloud Run instance with `4Gi` of memory and `2 vCPUs`, providing ample headroom for local ML processing.

## 7. The Qdrant Schema Mismatch
**The Problem:** Uploads began failing with `Wrong input: Not existing vector name error: sparse`.
**The Cause:** We upgraded the application to use Hybrid Search (Dense + Sparse), but the existing Qdrant collection (`askmynotes_global`) was created during Phase 1 when only Dense vectors existed. Qdrant rejected the sparse payloads.
**The Fix:** Rather than writing a complex database migration, we bumped the global collection name to `askmynotes_v2` in `main.py`. This forced Qdrant to spin up a fresh, perfectly configured collection on the very next upload.

## 8. The Empty Environment Variable Bug
**The Problem:** Final integration testing revealed that `ask` and `quiz` endpoints failed with: `The model '' does not exist`.
**The Cause:** The `GROQ_MODEL` environment variable existed in Cloud Run but was an empty string. `os.getenv("GROQ_MODEL", "fallback")` returns `""` if the variable exists but is empty, causing the LLM call to fail.
**The Fix:** Refactored the Python logic across the codebase to use strict boolean fallback: `os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"`.

## Conclusion
By meticulously peeling back these layers—from networking and routing to memory management, filesystem caches, and vector schemas—we successfully transformed a fragile local script into a highly robust, Serverless Machine Learning pipeline.
