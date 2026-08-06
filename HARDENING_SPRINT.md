# Hardening Sprint 2026 — Architecture & Operations Guide

Scope: Podcast Opportunity Miner (content-miner) + AI-Youtube-Shorts-Generator
hardening sprint. Fixes correctness, contract, evaluation, cache, timeline,
and boundary gaps before feature expansion.

## 1. Repository ownership (non-negotiable)

| Repository | Owns | Must NOT own |
|---|---|---|
| youtube-content-miner | Discovery, canonical transcript, opportunity scoring, candidate generation, boundary refinement, ranking, review state, analytics, render request construction. | Media rendering, re-ranking inside renderer, changing approved boundaries. |
| youtube-content-miner | — | — |
| AI-Youtube-Shorts-Generator | Source download/cache, alignment fallback, face/speaker tracking, camera plan, crop/layout, captions, audio, thumbnails, encoding, technical QC. | Choosing viral moments, changing boundaries, independent highlight ranking. |

Miner decides WHAT to clip; renderer decides HOW to present approved boundaries.

## 2. Job state machine (renderer)

```
queued       -> downloading | cancelled
downloading  -> analysing   | failed
analysing    -> rendering   | failed
rendering    -> quality_check | failed
quality_check-> completed | partial_failure | failed
terminal: completed, partial_failure, failed, cancelled, orphaned
```

- All transitions route through `transition_job(job_id, expected, target)`
  (compare-and-swap, atomic memory + SQLite under one lock). Exactly one
  queued transition wins (worker queued->downloading vs cancel
  queued->cancelled).
- Retry only from failed / partial_failure.
- Partial-failure exposes successful artifacts identically from memory and
  the persisted store.

## 3. Cache timeline correctness

- Rendered clip caches are keyed by source + time + aspect + resolution +
  RENDER PROFILE version (camera/caption/tracker/encoder).
- Each cached media has a `<key>.timeline.json` sidecar. On a cache hit the
  renderer returns the same typed `(path, RenderTimeline>` as a miss, loaded
  from the sidecar. A missing sidecar invalidates the entry — module-global
  state from another render is never read.

## 4. Contract (RenderRequestV2)

- content-miner builds the contract; the renderer validates it
  (JSON Schema + Zod + Pydantic on shared fixtures).
- `caption_plan` carries `language`, `provider`, `transcript_version`,
  `alignment_confidence`, `cues[].words[]` (canonical word timing).
  Trusted word timing bypasses full Whisper re-transcription; cue-only
  timing falls back to forced alignment against canonical text; no trusted
  transcript falls back to Whisper with provenance labelled.
- Cross-field invariants (cue-in-range, end>start, duplicate ids, narrative
  ordering, word-in-cue) are enforced by Zod and Pydantic (standard JSON
  Schema cannot reference sibling values).

## 5. Candidate lifecycle

Each segment has stable `candidate_id`, `generation_run_id`, `revision`,
`parent_candidate_id`, `boundary_source` (rough|semantic|repair|manual) and
`scoring_version`, propagated into the clips table (additive columns).

## 6. Start-boundary gate

`MID_SENTENCE`, `MISSING_CONTEXT`, `UNRESOLVED_REFERENCE` are repaired
(pull start back to a complete prior utterance) or rejected — never a soft
cap. `LATE_HOOK` stays a scoring penalty.

## 7. Operations

- Render service: `.venv/Scripts/python.exe render_service.py` (port 8084).
- Tests: miner `npx vitest run`; renderer
  `.venv/Scripts/python.exe -m pytest <files> -q`.
- Golden evaluation: `src/lib/golden/` fixtures + temporal-IoU metrics.