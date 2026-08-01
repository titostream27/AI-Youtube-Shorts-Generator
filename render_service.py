"""Render service for the hybrid miner integration.

The youtube-content-miner already does discovery + scoring, so this service
only renders: download the source video once (cached), cut each clip to its
[start, end] and reframe it vertically (9:16 by default). No LLM calls, no
transcription — pure ffmpeg + OpenCV work.

Run:
    .venv/Scripts/python.exe render_service.py

Endpoints:
    GET  /health
    POST /api/render   {video_url, clips: [{clip_id, title, start_sec, end_sec}], aspect_ratio}
"""
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from shorts_generator.local.clipper import crop_highlights_local
from shorts_generator.local.downloader import download_youtube_local

RENDER_ROOT = Path(os.getenv("RENDER_OUTPUT_DIR", "rendered")).resolve()
HOST = os.getenv("RENDER_HOST", "127.0.0.1")
PORT = int(os.getenv("RENDER_PORT", "8084"))
FORMAT = os.getenv("RENDER_FORMAT", "720")

app = FastAPI(title="Shorts Render Service", version="0.1.0")


class ClipRequest(BaseModel):
    clip_id: int | str
    title: str = ""
    start_sec: float
    end_sec: float


class RenderRequest(BaseModel):
    video_url: str
    clips: List[ClipRequest] = Field(min_length=1)
    aspect_ratio: str = "9:16"


class RenderResponse(BaseModel):
    job_id: str
    source_video: str
    rendered: List[Dict]


@app.get("/health")
def health():
    return {"status": "ok", "service": "shorts-render", "version": "0.1.0"}


@app.get("/files/{job_id}/{filename}")
def serve_file(job_id: str, filename: str):
    """Serve a rendered short so the miner UI can link / play it directly.

    Path traversal guard: both segments must be plain names, and the resolved
    path must stay inside the render root.
    """
    if not job_id or not filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid file path")
    path = (RENDER_ROOT / job_id / filename).resolve()
    root = RENDER_ROOT.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="video/mp4")


def _render(request: RenderRequest) -> RenderResponse:
    job_id = uuid.uuid4().hex[:10]
    job_dir = RENDER_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download once (cached by video id).
    try:
        source = download_youtube_local(
            request.video_url,
            fmt=FORMAT,
            out_dir=str(RENDER_ROOT / "source"),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"download failed: {e}") from e

    # 2. Render each clip as a vertical short.
    highlights = [
        {
            "title": c.title or f"clip-{c.clip_id}",
            "start_time": float(c.start_sec),
            "end_time": float(c.end_sec),
        }
        for c in request.clips
    ]

    start = time.time()
    results = crop_highlights_local(
        source,
        highlights,
        aspect_ratio=request.aspect_ratio,
        out_dir=str(job_dir),
    )

    rendered = []
    for c, r in zip(request.clips, results):
        item = {
            "clip_id": c.clip_id,
            "title": c.title,
            "start_sec": c.start_sec,
            "end_sec": c.end_sec,
            "status": "ok" if r.get("clip_url") else "error",
            "duration_sec": round(float(c.end_sec) - float(c.start_sec), 2),
        }
        if r.get("clip_url"):
            item["clip_path"] = os.path.abspath(r["clip_url"])
            # Browser-reachable path relative to the render root: <job>/<file>.
            item["clip_url"] = f"{job_id}/{os.path.basename(r['clip_url'])}"
        if r.get("error"):
            item["error"] = r["error"]
        rendered.append(item)

    return RenderResponse(
        job_id=job_id,
        source_video=source,
        rendered=rendered,
    )


@app.post("/api/render", response_model=RenderResponse)
def render(request: RenderRequest):
    """Render clips synchronously. Long videos download first — poll client-side."""
    return _render(request)


if __name__ == "__main__":
    import uvicorn

    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[render] output root: {RENDER_ROOT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
