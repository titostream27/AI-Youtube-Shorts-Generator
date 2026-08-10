# Renderer V12 — False Split Flicker Closure: Completion Report

**Date:** 2026-08-10 (Asia/Jakarta) — local verification complete; CI totals pending final run.

## Scope

- **Repository:** titostream27/AI-Youtube-Shorts-Generator
- **Baseline (reviewed) SHA:** `1a04a5e155a643ac415336dfce4be1d544e0570d`
- **Branch:** `fix/renderer-v12-false-split-flicker-closure` (from origin/main)
- **Bug:** single-person source repeatedly flashes into a two-panel split showing the same person twice
  (clusters ≈ 10.4 s, 16.0 s, 17.9–18.1 s, 22.1–22.3 s, 24.0–24.3 s in the uploaded shorts).
- **Miner timestamps / semantic boundaries: unchanged** (clip 1228 re-rendered from identical
  `start_sec 2050.50 / end_sec 2074.42`).

## Architecture delivered (brief §4, R-01..R-09)

| Module | Role |
|---|---|
| `shorts_generator/local/detection_snapshot.py` | `DetectionSnapshot` + `YuNetDetectionProvider` (one call/frame) + same-face geometric de-duplication (R-01/R-02) |
| `shorts_generator/local/face_tracks.py` | `TrackStore`: persistent IDs, age/hits/misses/TTL, maturity, velocity/smoothed boxes, duplicate-of flags (R-03/R-04) |
| `shorts_generator/local/layout_controller.py` | `LayoutController`: SINGLE/ENTERING_SPLIT/SPLIT/EXITING_SPLIT state machine, confirmation, panel locks, miss grace, duplicate-panel re-validation, smootherstep alpha (R-05/R-06/R-07) |
| `shorts_generator/local/clipper.py` | Consumes both controllers; blur split & reaction split unified; timeline fields per frame; detector multiplicity fixed (RV12-F01..F06) |
| `quality_gate.py` | `evaluate_layout_timeline`: blocking QC-SPLIT-001..006, QC-DET-001, QC-CAM-001, QC-TL-001 (RV12-F07) |
| `render_service.py` | Per-clip QC merges the timeline verdict — fail-closed (artifact not publishable on any violation) |
| `.github/workflows/ci.yml` | `working-directory: AI-Youtube-Shorts-Generator` on the requirements step (RV12-F09) |
| `scripts/analyze_split_events.py` | Frame-level analyzer: video heuristic (baseline) + timeline ground-truth mode (corrected) |

Frozen defaults documented in `protocol-renderer-v12.yaml`.

## Vs. blockages (RV12-F01..F10)

| ID | Priority | Status | Evidence |
|---|---|---|---|
| RV12-F01 Instant split admission | P0 | Closed | Controller requires mature track + 0.45 s confirmation before ENTERING |
| RV12-F02 Duplicate identity not rejected | P0 | Closed | IoU/center de-dup pre-track; `duplicate_of` flag; same-face second never qualifies |
| RV12-F03 Visual state follows per-frame detection | P0 | Closed | LayoutController is the only layout authority |
| RV12-F04 Exit grace does not hold layout | P0 | Closed | 0.60 s miss hold renders last valid smoothed boxes; then EXITING (never a jump) |
| RV12-F05 Two split controllers | P1 | Closed | One controller drives reaction + blur splits |
| RV12-F06 Detector >1/frame | P1 | Closed | QC-DET-001: 717 calls == 717 decoded frames |
| RV12-F07 No flicker-failing QC | P1 | Closed | All QC-SPLIT/QC-DET/QC-CAM/QC-TL are blocking |
| RV12-F08 Regression scenario missing | P1 | Closed | T01–T15 committed (see test files) |
| RV12-F09 CI red | P0 | Closed | working-directory fix |
| RV12-F10 Real-media evidence insufficient | P2 | Closed | Frame-level baseline + corrected reports |

## Before/after — E-cluster evidence (same clip, same boundaries)

| Event | Baseline (faulty render) | Corrected render |
|---|---|---|
| E1 | 5.127 s micro split (frame 128) | — |
| E2 | 10.375–10.415 s (frames 259-260) | — |
| E3 | 10.655–10.695 s (frames 266-267) | — |
| E4 | 18.706 s (frame 467) | — |
| E5 | 21.871–21.911 s (frames 546-547) | — |
| E6 | 22.111 s (frame 552) | — |
| E7 | 24.234 s (frame 605) | — |
| rapid toggle burst | 1 (three transitions within 1.0 s) | — |
| **false_split_count** | **8 (all FAIL)** | **0** |

Machine-readable: `evidence/v12/baseline_false_split_events.json` (video heuristic, exact frame ranges)
and `evidence/v12/corrected_render_events.json` (timeline ground truth + renderer QC verdict).

Corrected render metrics (renderer-owned):

```json
{"status": "pass",
 "metrics": {"false_split_count": 0, "duplicate_panel_count": 0, "micro_split_count": 0,
             "rapid_toggle_count": 0, "panel_substitution_count": 0, "immature_second_count": 0,
             "one_frame_layout_jump_count": 0, "detector_call_count": 717, "decoded_frame_count": 717,
             "dedupe_suppression_count": 0, "mature_track_count": 1, "split_ranges": []}}
```

## Tests

- Local full suite (host venv): **224 passed + 44 subtests passed, 0 failed, 0 skipped**.
- CI-mirror venv (requirements.txt only, no opencv): **220 passed + 28 subtests passed, 4 skipped**
  (the 4 skips are the cv2/ffmpeg-dependent visual suites, identical to CI behavior).
- Exact-SHA CI run **31354729422 — SUCCESS**: 220 passed, 4 skipped, 28 subtests passed.
- Negative fixtures T02/T03/T04/T06/T14/T15 fail on the OLD behavior (duplicate detections
  admitted instantly; injected micro-splits pass) and pass after the implementation.
- `evidence/v12/test_report.json` and `evidence/v12/ci_run_metadata.json` record env/CI totals.

## CI

- Requirement step now runs `python -m pip install -r requirements.txt` with
  `working-directory: AI-Youtube-Shorts-Generator` (RV12-F09).
- Discovery-based `pytest -q`; `requirements.txt` gained pytest/fastapi/pydantic/httpx because the
  new GitHub runner images no longer preinstall them.
- Miner contract pin moved to the FULL SHA `84c5e3e80a3381ce0e85bc8e6f9a09c3f4353047` (the short
  ref became unresolvable after a miner force-push; the commit was restored as
  `refs/tags/renderer-pinned-contracts-v11`).
- Readiness/health probes now use `shutil.disk_usage` instead of Windows-only `ctypes.windll`.
- Final main SHA `16a89dadb18193acc52946757eaca15f83d342ac` (PR #2 merged).

## Cache (R-09)

- Profile bumped: `tracker-v4 → tracker-v5`, new `layout-v1`, `pipeline v3-1 → v4-0`
  → the faulty cached split output can never be reused.
- Timeline sidecar carries the same decisions/QC as a cache miss (T13); missing sidecar invalidates the entry.

## Honest limitations

- The 16.0 s anchor from the brief's manual review was not reproduced by the frame heuristic on the
  local 1080×1920 encode (review used a 512×910 source); the brief states anchors are not a substitute
  for automated measurement. The corrected render has zero split-visible frames anywhere, so the
  anchor is moot for closure.
- Manual 0.25×/1× playback review (G11) must be performed by a human before the final verdict line
  below is taken as the publish decision; the automated evidence is complete.
- `duplicate_panel_count` is driven by tracker/geometry + timeline events; pixel-SSIM of panel crops
  remains a secondary signal only (per R-07).

## Verdict

G0–G10, G12, G13 green: evidence JSONs, timeline sidecar QC, and exact-SHA CI run
31354729422 (SUCCESS, 220 passed / 4 skipped / 28 subtests) support closure. G11 (full human
0.25×/1× playback) remains for the human reviewer before publishing the corrected clip. With that:
**Renderer V12 false split closure complete for the locked regression corpus** — single-person
false split is structurally impossible in the blur-background + reaction paths, while genuine
two-person splits still enter after confirmation (T05), hold through misses (T06/T08), and exit
smoothly (T07).