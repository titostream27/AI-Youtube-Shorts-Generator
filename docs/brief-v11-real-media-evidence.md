# Brief V11 Renderer Evidence — Corrected Recovery State

**Date:** 2026-08-08
**Verdict:** **BLOCKED**

## Local renderer verification

- CI-parity full discovery: **205 passed, 44 subtests passed, 0 failed, 0 errors, 0 skipped**.
- Required visual regression: **4 passed, 16 subtests passed, 0 skipped**.
- Before the recovery patch the same visual suite produced four skips with exit 0 because OpenCV was absent. CI now installs `opencv-python-headless`, generates deterministic fixtures, imports `cv2` fail-fast, and runs the suite.

## Historical real MP4s

Historical captioned outputs exist for Iqbal, Kobe, and Kim. `ffprobe` confirms H.264, 1080×1920, yuv420p, AAC audio. They are not all valid current G3 evidence:

- Iqbal matches the current post-fix production-selected accepted clip.
- Kobe and Kim do not pass the current post-fix production selection threshold.
- No current evidence proves two genuine speaker-switch selections.
- No complete PASS/FAIL/N/A full-playback checklist exists for three qualified outputs.

Synthetic six-second files under `evidence_out/` are visual-test fixtures and are not counted as real-media G3 evidence.

## Remote CI

Final recovery SHA `f3e7f7d70fdbe15f09cc77bd33b67f1a520af605` is green on GitHub Actions run `31253140819`.

Failures diagnosed and fixed on the way to green:

- `31252188799` (d8d8a8e): checkout could not fetch the short miner ref `84c5e3e` — pinned the full immutable SHA.
- `31252990091` (034ec1a): pytest collection failed because fastapi/pydantic were not installed, visual tests failed without ffmpeg, and `/readyz` used a Windows-only `ctypes.windll` free-disk check. Fixed by adding the runtime/test dependencies, installing ffmpeg in CI, and switching both free-disk checks to cross-platform `shutil.disk_usage`.

Local CI-parity full discovery with only `requirements.txt` + fixture generation: **205 passed, 44 subtests passed, 0 failed, 0 errors, 0 skipped**.

## G3 verdict

**BLOCKED:** fewer than three currently production-selected qualified outputs, fewer than two proven genuine speaker-switch cases, and incomplete full-playback review.
