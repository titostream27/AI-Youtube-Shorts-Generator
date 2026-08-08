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

The latest pushed fork run inspected was red and predates this recovery patch. Renderer CI is BLOCKED until the final recovery SHA is pushed and its exact GitHub Actions run is green.

## G3 verdict

**BLOCKED:** fewer than three currently production-selected qualified outputs, fewer than two proven genuine speaker-switch cases, and incomplete full-playback review.
