"""Local YouTube download via yt-dlp.

Returns a local mp4 path so the rest of the local pipeline can read it
directly off disk.

Phase 2 (Stability) upgrades per the upgrade brief §37-38:
  - format selector supports up to 2160p with VP9/AV1 fallbacks (not mp4-only)
  - resolution-aware cache: source_<id>_720p.mp4 / _1080p.mp4 / _1440p.mp4 /
    _2160p.mp4; a cached file below the requested height is rejected and
    re-downloaded, with an explicit log line explaining the decision.
"""
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

from ..config import LOCAL_OUTPUT_DIR


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str) -> str:
    """Map our '720'/'1080'/'1440'/'2160' shorthand to a yt-dlp format selector.

    The brief (§37) requires supporting VP9/AV1 for 1440p/2160p — restricting
    to mp4 only would force a 1080p fallback on most high-res sources. We still
    prefer mp4 when available, but allow vp9/av1 video with any audio container
    and let yt-dlp merge.
    """
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}][ext=mp4]/"
        f"best[height<={height}]/"
        f"18/best"
    )


def _extract_youtube_video_id(source: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video id from a URL."""
    parsed = urlparse(source)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
            return video_id or None
        match = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", parsed.path)
        if match:
            return match.group(1)

    return None


def _resolve_local_path(source: str) -> Optional[str]:
    """Return a local filesystem path if the input already points at one."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raw_path = f"//{parsed.netloc}{raw_path}"
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        raise RuntimeError(f"Local file URL does not exist: {source}")

    if parsed.scheme in ("http", "https"):
        return None

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    if any(sep in source for sep in (os.sep, "/")) or source.startswith("~") or source.startswith("."):
        raise RuntimeError(f"Local file path does not exist: {source}")

    return None


def _probe_height(path: str) -> Optional[int]:
    """Return the video stream height (px) of a file via ffprobe, or None."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return int(probe.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    return None


def _has_audio(path: str) -> bool:
    """True if the file has at least one audio stream."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20,
        )
        return probe.returncode == 0 and bool(probe.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _existing_download(out_dir: str, video_id: str, fmt: str) -> Optional[str]:
    """Return a resolution-appropriate cached download, or None.

    Phase 2 (§38): the cache key now includes the requested height so a 720p
    download is never reused for a 2160p request. Each candidate is validated:
    it must exist, have audio, and its probed height must be >= the requested
    height. Anything lower is logged as rejected and ignored.
    """
    try:
        height = int(fmt)
    except ValueError:
        height = 0  # unknown fmt -> any cached file is acceptable

    # Candidates: resolution-specific first, then the legacy generic name.
    candidates = []
    if height > 0:
        for ext in (".mp4", ".mkv", ".webm"):
            candidates.append(os.path.join(out_dir, f"source_{video_id}_{height}p{ext}"))
    for ext in (".mp4", ".mkv", ".webm"):
        candidates.append(os.path.join(out_dir, f"source_{video_id}{ext}"))

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if not _has_audio(candidate):
            print(f"[download/local] cached file has no audio, removing: {candidate}", flush=True)
            try:
                os.remove(candidate)
            except OSError:
                pass
            continue
        cached_h = _probe_height(candidate)
        if height > 0 and cached_h is not None and cached_h < height:
            print(
                f"[download/local] Cached source: {cached_h}p | Requested source: up to {height}p | "
                f"Cache rejected: resolution below request ({os.path.basename(candidate)})",
                flush=True,
            )
            continue
        print(f"[download/local] reusing cached download: {candidate}", flush=True)
        return candidate
    return None


def download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None) -> str:
    """Download a remote URL or return a local file path unchanged."""
    local_path = _resolve_local_path(video_url)
    if local_path:
        print(f"[download/local] using local file: {local_path}", flush=True)
        return local_path

    yt_dlp = _import_ytdlp()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    video_id = _extract_youtube_video_id(video_url)
    if video_id:
        cached = _existing_download(out_dir, video_id, fmt)
        if cached:
            return cached

    try:
        height = int(fmt)
    except ValueError:
        height = 720
    # Resolution-aware output template: source_<id>_<height>p.<ext>.
    outtmpl = os.path.join(out_dir, f"source_%(id)s_{height}p.%(ext)s")

    print(f"[download/local] {video_url} @ {height}p -> {out_dir}/", flush=True)
    ydl_opts = {
        "format": _format_for(fmt),
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        path = ydl.prepare_filename(info)
        # merge_output_format may rename the extension after merge
        if not os.path.exists(path):
            stem, _ = os.path.splitext(path)
            for ext in (".mp4", ".mkv", ".webm"):
                if os.path.exists(stem + ext):
                    path = stem + ext
                    break

    print(f"[download/local] ready: {path}", flush=True)
    return path
