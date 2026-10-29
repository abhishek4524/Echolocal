"""
api/server.py
EchoLocal REST API — FastAPI server for programmatic dubbing pipeline access.

Endpoints:
    POST /api/v1/dub          Submit a dubbing job (file upload or YouTube URL)
    GET  /api/v1/status/{id}  Poll job status
    GET  /api/v1/download/{id} Stream finished video
    GET  /api/v1/jobs         List all jobs
    DELETE /api/v1/jobs/{id}  Delete a job and its artifacts

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker-compose up api
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# ── FastAPI & pydantic ────────────────────────────────────────────────────────
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoLocal API",
    description=(
        "AI-powered video dubbing pipeline — transcribe, translate, "
        "voice-clone and lip-sync any video locally."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Thread pool for CPU/GPU pipeline work ─────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="echolocal-worker")

# ── In-memory job store (swap for Redis/SQLite in production) ─────────────────
_jobs: Dict[str, "JobState"] = {}

JOBS_DIR = PROJECT_ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data models
# ═════════════════════════════════════════════════════════════════════════════

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobState(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    source: str = ""          # filename or YouTube URL
    target_lang: str = "hi"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    progress_pct: int = 0
    current_step: str = "pending"
    output_path: Optional[str] = None
    error: Optional[str] = None
    duration_s: Optional[float] = None


class SubmitResponse(BaseModel):
    job_id: str
    message: str
    status_url: str
    download_url: str


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    source: str
    target_lang: str
    created_at: str
    updated_at: str
    progress_pct: int
    current_step: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    duration_s: Optional[float] = None


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline runner (runs in thread pool)
# ═════════════════════════════════════════════════════════════════════════════

def _update_job(job_id: str, **kwargs):
    """Thread-safe job state update helper."""
    job = _jobs.get(job_id)
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = datetime.utcnow().isoformat()


def _run_pipeline(job_id: str, video_path: Path, target_lang: str) -> None:
    """
    Execute the full EchoLocal pipeline synchronously in a worker thread.
    Imports are deferred so the API server starts fast even without GPU.
    """
    import time
    start = time.perf_counter()

    try:
        _update_job(job_id, status=JobStatus.RUNNING, current_step="extracting audio", progress_pct=5)
        from extract_audio import extract_audio
        audio_path = video_path.parent / (video_path.stem + "_audio.wav")
        extract_audio(str(video_path), str(audio_path))

        _update_job(job_id, current_step="separating stems", progress_pct=15)
        from separate_audio import separate_audio
        stems = separate_audio(str(audio_path), str(video_path.parent))

        _update_job(job_id, current_step="transcribing", progress_pct=25)
        from transcribe import transcribe_audio
        segments = transcribe_audio(stems.get("vocals", str(audio_path)))

        _update_job(job_id, current_step="diarizing speakers", progress_pct=35)
        from diarize import diarize_audio
        segments = diarize_audio(stems.get("vocals", str(audio_path)), segments)

        _update_job(job_id, current_step="translating", progress_pct=45)
        from translate import translate_segments
        translated = translate_segments(segments, target_lang=target_lang)

        _update_job(job_id, current_step="generating subtitles", progress_pct=50)
        from generate_subtitles import generate_srt
        srt_path = video_path.parent / (video_path.stem + ".srt")
        generate_srt(translated, str(srt_path))

        _update_job(job_id, current_step="cloning voices & TTS", progress_pct=55)
        from voice_clone_tts import generate_dubbed_audio
        dubbed_audio = generate_dubbed_audio(
            translated,
            stems.get("vocals", str(audio_path)),
            str(video_path.parent)
        )

        _update_job(job_id, current_step="merging audio", progress_pct=75)
        from merge_final import merge_audio
        final_audio = merge_audio(
            dubbed_audio,
            stems.get("no_vocals", ""),
            str(video_path.parent)
        )

        _update_job(job_id, current_step="muxing final video", progress_pct=88)
        from merge_final import mux_video
        out_dir = PROJECT_ROOT / "output"
        out_dir.mkdir(exist_ok=True)
        final_video = out_dir / (video_path.stem + "_dubbed.mp4")
        mux_video(str(video_path), final_audio, str(final_video))

        elapsed = round(time.perf_counter() - start, 2)
        _update_job(
            job_id,
            status=JobStatus.DONE,
            progress_pct=100,
            current_step="complete",
            output_path=str(final_video),
            duration_s=elapsed,
        )
        logger.info(f"Job {job_id} completed in {elapsed}s → {final_video}")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
        _update_job(job_id, status=JobStatus.FAILED, current_step="failed", error=str(exc))
    finally:
        # Clean up temp job workspace
        try:
            shutil.rmtree(video_path.parent, ignore_errors=True)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "EchoLocal API v1.0.0 — see /docs"}


@app.post("/api/v1/dub", response_model=SubmitResponse, status_code=202)
async def submit_job(
    background_tasks: BackgroundTasks,
    target_lang: str = Form("hi", description="Target language code (e.g. 'hi', 'en', 'fr')"),
    youtube_url: Optional[str] = Form(None, description="YouTube URL to dub"),
    video_file: Optional[UploadFile] = File(None, description="Video file to dub"),
):
    """
    Submit a dubbing job.

    Provide either `youtube_url` or `video_file` (not both).
    Returns a `job_id` for polling status.
    """
    if not youtube_url and not video_file:
        raise HTTPException(status_code=422, detail="Provide either 'youtube_url' or 'video_file'.")
    if youtube_url and video_file:
        raise HTTPException(status_code=422, detail="Provide only one: 'youtube_url' or 'video_file'.")

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    if youtube_url:
        # Download YouTube video into job workspace
        source_label = youtube_url
        async def _prepare():
            from download_youtube import download_youtube_video
            return download_youtube_video(youtube_url, output_dir=job_dir)
        try:
            video_path = await asyncio.get_event_loop().run_in_executor(_executor, lambda: __import__("download_youtube", fromlist=["download_youtube_video"]).download_youtube_video(youtube_url, output_dir=job_dir))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YouTube download failed: {e}")
    else:
        source_label = video_file.filename or "upload.mp4"
        video_path = job_dir / source_label
        with open(video_path, "wb") as f:
            content = await video_file.read()
            f.write(content)

    job = JobState(job_id=job_id, source=source_label, target_lang=target_lang)
    _jobs[job_id] = job

    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        _executor,
        _run_pipeline,
        job_id,
        video_path,
        target_lang,
    )

    logger.info(f"Job {job_id} submitted — source: {source_label}")

    base = "/api/v1"
    return SubmitResponse(
        job_id=job_id,
        message="Job queued successfully.",
        status_url=f"{base}/status/{job_id}",
        download_url=f"{base}/download/{job_id}",
    )


@app.get("/api/v1/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Poll the status and progress of a dubbing job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return StatusResponse(**job.dict())


@app.get("/api/v1/download/{job_id}")
async def download_result(job_id: str):
    """Download the dubbed video once the job status is 'done'."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job not complete yet (status: {job.status}).")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=500, detail="Output file missing.")
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=Path(job.output_path).name,
    )


@app.get("/api/v1/jobs", response_model=List[StatusResponse])
async def list_jobs():
    """List all jobs (most recent first)."""
    return sorted(
        [StatusResponse(**j.dict()) for j in _jobs.values()],
        key=lambda x: x.created_at,
        reverse=True,
    )


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Delete a job and its output file."""
    job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.output_path:
        try:
            Path(job.output_path).unlink(missing_ok=True)
        except Exception:
            pass
    return JSONResponse(status_code=204, content=None)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "active_jobs": sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING),
        "total_jobs": len(_jobs),
    }
