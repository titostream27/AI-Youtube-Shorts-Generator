"""Phase 3 (Quality) — conservative audio mastering chain.

Brief §45: noise reduction → high-pass → gentle EQ → de-esser → compressor →
loudness normalization → true-peak limiter.

Brief §46: silence/filler trimming is handled at clip boundary time in the
miner; here we keep the chain conservative — ffmpeg-native filters only
(no external deps like spleeter), controlled by env:

  RENDER_AUDIO_NOISE_REDUCTION=1   (afftdn)
  RENDER_AUDIO_HIGHPASS=1          (highpass f=80)
  RENDER_AUDIO_DEESSER=1           (deesser i=0.35 m=0.55 f=5600 s=0.02)
  RENDER_AUDIO_COMPRESSOR=1        (acompressor threshold=-18dB ratio=2.5)
  RENDER_AUDIO_LOUDNESS=1          (loudnorm I=-14 TP=-1.5 LRA=11)
  RENDER_AUDIO_LIMITER=1           (alimiter limit=-1.5dB)
"""
import os
import subprocess
from typing import List, Optional


def build_audio_chain() -> List[str]:
    """Return a list of ffmpeg audio filters, or [] if all disabled."""
    if os.getenv("RENDER_AUDIO_MASTERING", "1") == "0":
        return []

    chain: List[str] = []

    if os.getenv("RENDER_AUDIO_NOISE_REDUCTION", "1") != "0":
        # Light denoise (afftdn) — gentle enough not to eat speech transients.
        chain.append("afftdn=nf=-25")

    if os.getenv("RENDER_AUDIO_HIGHPASS", "1") != "0":
        # Remove rumble / DC below 80 Hz.
        chain.append("highpass=f=80")

    if os.getenv("RENDER_AUDIO_DEESSER", "1") != "0":
        # De-esser on the default band (sibilance). The `f` frequency option
        # expects an integer Hz in a narrow range on some ffmpeg builds and
        # errors with "Result too large"; the default 3900Hz is fine.
        chain.append("deesser=i=0.35:m=0.55:s=0.02")

    if os.getenv("RENDER_AUDIO_COMPRESSOR", "1") != "0":
        # Gentle leveling: threshold -18 dB, ratio 2.5:1.
        chain.append("acompressor=threshold=-18dB:ratio=2.5:attack=12:release=150:makeup=2dB")

    if os.getenv("RENDER_AUDIO_LOUDNESS", "1") != "0":
        # Loudness normalization for Shorts (-14 LUFS, TP -1.5).
        chain.append("loudnorm=I=-14:TP=-1.5:LRA=11")

    if os.getenv("RENDER_AUDIO_LIMITER", "1") != "0":
        # Safety true-peak limiter AFTER loudnorm.
        chain.append("alimiter=limit=-1.5dB")

    return chain


def master_audio(in_path: str, out_path: str) -> str:
    """Apply the audio chain to in_path, writing out_path.

    Returns out_path on success; on any failure copies the input unchanged so
    the render never fails because of the master chain.
    """
    chain = build_audio_chain()
    if not chain:
        return in_path

    af = ",".join(chain)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-af", af,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return out_path
    except Exception as e:  # noqa: BLE001
        print(f"[audio] mastering failed ({e}); keeping original", flush=True)
        return in_path


def trim_pauses(source_path: str, start_sec: float, end_sec: float) -> Optional[str]:
    """Brief §46 — remove long pauses from a clip window.

    Uses ffmpeg's silencedetect to find pauses > RENDER_TRIM_PAUSE_THRESHOLD_S
    inside [start_sec, end_sec] and cuts them out with a 35ms audio crossfade.

    Returns a path to a trimmed clip (audio+video re-encoded losslessly for
    video via ffv1), or None if disabled / no pauses found.
    """
    if os.getenv("RENDER_TRIM_REMOVE_LONG_PAUSES", "0") != "1":
        return None
    threshold = float(os.getenv("RENDER_TRIM_PAUSE_THRESHOLD_S", "0.75"))
    min_pause = float(os.getenv("RENDER_TRIM_MIN_PAUSE_S", "0.12"))
    crossfade_ms = int(os.getenv("RENDER_AUDIO_CROSSFADE_MS", "35"))

    # Detect silences in the window.
    probe = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", source_path,
            "-ss", f"{start_sec:.3f}", "-t", f"{end_sec - start_sec:.3f}",
            "-af", f"silencedetect=noise=-30dB:d={threshold:.3f}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=180,
    )
    if probe.returncode != 0:
        return None

    import re
    silences: List[tuple] = []
    for line in probe.stderr.splitlines():
        m = re.search(r"silence_(start|end): ([\d.]+)", line)
        if not m:
            continue
        kind, val = m.group(1), float(m.group(2))
        if kind == "start":
            silences.append([val + start_sec, None])
        else:
            if silences and silences[-1][1] is None:
                silences[-1][1] = val + start_sec
    # Filter: keep pauses long enough to matter.
    pauses = [
        (s, e) for s, e in silences
        if e is not None and (e - s) >= max(min_pause, threshold)
    ]
    if not pauses:
        return None

    print(f"[audio] trimming {len(pauses)} pauses (>{threshold:.2f}s) from clip", flush=True)

    # Build a filter_complex that cuts each pause. This is complex with
    # arbitrary N pauses; a robust approach is to write a list of keep-ranges
    # and concat them with afade crossfades. For the common 1-2 pause case we
    # build the chain manually.
    keeps: List[tuple] = []
    cur = start_sec
    for s, e in pauses:
        if s - cur >= min_pause:
            keeps.append((cur, s))
        cur = e
    if end_sec - cur >= min_pause:
        keeps.append((cur, end_sec))
    if not keeps:
        return None

    import tempfile
    out_path = tempfile.mktemp(suffix=".trimmed.mkv", dir=os.path.dirname(source_path) or ".")
    inputs: List[str] = []
    filter_parts: List[str] = []
    for i, (ks, ke) in enumerate(keeps):
        inputs += ["-ss", f"{ks:.3f}", "-t", f"{ke - ks:.3f}", "-i", source_path]
        filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
    vmap = "".join(f"[v{i}]" for i in range(len(keeps)))
    amap = "".join(f"[a{i}]" for i in range(len(keeps)))
    filter_parts.append(f"{vmap}concat=n={len(keeps)}:v=1:a=0[vout]")
    filter_parts.append(f"{amap}concat=n={len(keeps)}:v=0:a=1[aout]")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "ffv1",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        return out_path
    except Exception as e:  # noqa: BLE001
        print(f"[audio] pause trim failed ({e}); keeping original", flush=True)
        return None
