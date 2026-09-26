"""
api/app.py

FastAPI service wrapping pipeline.run_analysis() so the UI (or curl/Swagger
at /docs) can upload a video and get back an annotated video + heatmaps,
instead of using main.py's CLI-only entry point.

Scope note: this is dev/demo scope for the course submission, not a
production deployment —
  - Job state lives in an in-memory dict (JOBS): lost on server restart,
    not shared across multiple worker processes.
  - One background thread per upload, no queue/concurrency limit. Fine for
    a single demo user; would need a real task queue (Celery/RQ) or at
    least a semaphore for multiple concurrent users on shared hardware.

Run with: uvicorn api.app:app --reload --port 8000
"""

import os
import shutil
import threading
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from configs import OUTPUT_ROOT_DIR, DEFAULT_PLAYER_MODEL, PLAYER_MODEL_REGISTRY
from pipeline import run_analysis

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

app = FastAPI(title="Basketball Player Tracking & Heatmap API")

# Dev CORS — the UI runs on a different port (npm run dev, :3000) than this
# API (:8000), so the browser blocks requests without this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves output/run_N/... directly, e.g. output/run_3/video_1_output.mp4
# becomes downloadable at /output/run_3/video_1_output.mp4
app.mount("/output", StaticFiles(directory=OUTPUT_ROOT_DIR), name="output")

# job_id -> {"status": "pending"|"processing"|"done"|"error", ...}
JOBS: dict[str, dict] = {}


def _to_public_path(path: str) -> str:
    """Rewrites a run_analysis() filesystem path (may use OS-native
    separators) into a URL path served under the /output mount."""
    rel = os.path.relpath(path, OUTPUT_ROOT_DIR)
    return "/output/" + rel.replace(os.sep, "/")


def _process_job(job_id: str, video_path: str, player_model: str) -> None:
    JOBS[job_id]["status"] = "processing"
    try:
        run_info = run_analysis(video_path, player_model=player_model)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            **run_info,
            "output_video_url": _to_public_path(run_info["output_video"]),
            "heatmap_dir_url": _to_public_path(run_info["heatmap_dir"]),
            "player_heatmaps": [
                {"player_id": h["player_id"], "url": _to_public_path(h["path"])}
                for h in run_info["player_heatmaps"]
            ],
        }
    except Exception as exc:  # noqa: BLE001 - report any failure to the client instead of crashing the thread silently
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)


@app.get("/models")
async def list_models() -> dict:
    """Player-detector architectures the UI can let the user pick between
    before running /analyze — real mAP numbers behind each label live in
    REPORT.md §3 (ablation), not placeholders. Only id/label/description are
    exposed — path_base is an internal filesystem detail, not API metadata."""
    return {
        "default": DEFAULT_PLAYER_MODEL,
        "models": [
            {"id": model_id, "label": entry["label"], "description": entry["description"]}
            for model_id, entry in PLAYER_MODEL_REGISTRY.items()
        ],
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    player_model: str = Form(DEFAULT_PLAYER_MODEL),
) -> dict:
    """Uploads a video and starts analysis in the background. Returns a
    job_id immediately — poll GET /jobs/{job_id} for progress/result."""
    if player_model not in PLAYER_MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown player_model '{player_model}'. See GET /models.")

    job_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "source_filename": file.filename,
        "player_model": player_model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    thread = threading.Thread(target=_process_job, args=(job_id, video_path, player_model), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs")
async def list_jobs() -> list:
    return list(JOBS.values())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
