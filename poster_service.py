"""YouTube poster service — upload rendered shorts to YouTube.

FastAPI service on port 8085. Receives a publish request (clip metadata +
the render job's file URL), downloads the mp4, and uploads it to YouTube
via the Data API v3 (videos.insert, resumable upload).

Credentials are read from .env (YT_CLIENT_ID / YT_CLIENT_SECRET /
YT_REFRESH_TOKEN) or process env. The refresh token decides WHICH YouTube
account receives the upload — swap it to post as a different account.

Endpoints:
  GET  /health
  POST /api/publish   body: { clip_id, title, description, tags, file_url }
"""
import os
import sys
from pathlib import Path
from typing import List, Optional

import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = int(os.getenv("POSTER_PORT", "8085"))
VIDEO_ROOT = Path(os.getenv("RENDER_OUTPUT_DIR", "rendered")).resolve()
CLIENT_ID = os.getenv("YT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("YT_CLIENT_SECRET", "")
REFRESH_TOKEN = os.getenv("YT_REFRESH_TOKEN", "")

# Load .env if present (same file the miner / render service use).
def _load_dotenv(paths: List[str]) -> None:
    global CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "YT_CLIENT_ID" and not CLIENT_ID:
                CLIENT_ID = v
            elif k == "YT_CLIENT_SECRET" and not CLIENT_SECRET:
                CLIENT_SECRET = v
            elif k == "YT_REFRESH_TOKEN" and not REFRESH_TOKEN:
                REFRESH_TOKEN = v
        if CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN:
            break


_here = os.path.dirname(os.path.abspath(__file__))
_load_dotenv([
    os.path.join(_here, ".env"),
    os.path.join(_here, "..", "content-miner", ".env"),
    os.path.join(_here, "..", "..", "content-miner", ".env"),
])


class PosterConfigError(Exception):
    pass


def _credentials() -> google.oauth2.credentials.Credentials:
    if not (CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN):
        raise PosterConfigError(
            "YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN must be set in .env"
        )
    creds = google.oauth2.credentials.Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _resolve_file(file_url: str) -> str:
    """Resolve a file URL to a local path.

    Accepts:
      - an absolute/relative local path (C:\\... or rendered/...)
      - an http(s) URL to the render service's /files endpoint
    """
    if file_url.startswith("http://") or file_url.startswith("https://"):
        import shutil
        import tempfile
        import urllib.request

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        with urllib.request.urlopen(file_url, timeout=300) as resp, open(tmp.name, "wb") as out:
            shutil.copyfileobj(resp, out)
        return tmp.name
    p = Path(file_url)
    if not p.is_absolute():
        p = VIDEO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"video file not found: {p}")
    return str(p)


def _upload_to_youtube(
    video_path: str,
    *,
    title: str,
    description: str,
    tags: List[str],
    privacy: str = "private",
) -> dict:
    creds = _credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": [t[:100] for t in tags[:500]],
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  upload progress: {int(status.progress() * 100)}%", flush=True)

    video_id = response.get("id", "")
    return {
        "videoId": video_id,
        "url": f"https://youtu.be/{video_id}",
        "title": response.get("snippet", {}).get("title", ""),
        "privacy": response.get("status", {}).get("privacyStatus", ""),
    }


def _set_thumbnail(video_id: str, thumbnail_path: str) -> bool:
    """Upload a custom thumbnail for a video via the thumbnails.set API.

    The video must be owned by the authenticated account; the thumbnail file
    is a JPG from the render service's /files endpoint or a local path.
    Returns True on success, False on failure (non-fatal for publish).
    """
    try:
        creds = _credentials()
        youtube = build("youtube", "v3", credentials=creds)
        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        print(f"[poster] thumbnail set for {video_id}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[poster] thumbnail failed ({e}), continuing", flush=True)
        return False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Shorts Poster", version="0.1.0")

    class PublishRequest(BaseModel):
        clip_id: int
        title: str
        description: str = ""
        tags: List[str] = []
        file_url: str
        thumbnail_url: str = ""
        privacy: str = "private"

    class PublishResponse(BaseModel):
        clip_id: int
        status: str
        videoId: Optional[str] = None
        url: Optional[str] = None
        title: Optional[str] = None
        privacy: Optional[str] = None
        error: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "shorts-poster", "version": "0.1.0"}

    @app.post("/api/publish", response_model=PublishResponse)
    async def publish(req: PublishRequest):
        try:
            video_path = _resolve_file(req.file_url)
            result = _upload_to_youtube(
                video_path,
                title=req.title,
                description=req.description,
                tags=req.tags,
                privacy=req.privacy,
            )
            # Custom thumbnail (non-fatal if it fails).
            thumbnail_set = False
            if req.thumbnail_url and result.get("videoId"):
                try:
                    thumb_path = _resolve_file(req.thumbnail_url)
                    thumbnail_set = _set_thumbnail(result["videoId"], thumb_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[poster] thumbnail resolve failed ({e})", flush=True)
            return PublishResponse(
                clip_id=req.clip_id,
                status="published",
                videoId=result["videoId"],
                url=result["url"],
                title=result["title"],
                privacy=result["privacy"],
                error=None if thumbnail_set else "thumbnail not set",
            )
        except PosterConfigError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except HttpError as e:
            detail = f"YouTube API error {e.resp.status}: {e.error.get('message', str(e)) if isinstance(e.error, dict) else e}"
            raise HTTPException(status_code=502, detail=detail) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"publish failed: {e}") from e

    class DeleteRequest(BaseModel):
        video_id: str

    class DeleteResponse(BaseModel):
        video_id: str
        status: str
        error: Optional[str] = None

    @app.post("/api/videos/delete", response_model=DeleteResponse)
    async def delete_video(req: DeleteRequest):
        """Delete a video from the connected YouTube channel (videos.delete)."""
        try:
            creds = _credentials()
            youtube = build("youtube", "v3", credentials=creds)
            youtube.videos().delete(id=req.video_id).execute()
            print(f"[poster] deleted video {req.video_id}", flush=True)
            return DeleteResponse(video_id=req.video_id, status="deleted")
        except HttpError as e:
            detail = f"YouTube API error {e.resp.status}: {e.error.get('message', str(e)) if isinstance(e.error, dict) else e}"
            raise HTTPException(status_code=502, detail=detail) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"delete failed: {e}") from e

except ImportError:
    # Minimal fallback so the module still imports for credential checks.
    app = None


if __name__ == "__main__":
    import uvicorn

    print(f"[poster] port {PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
