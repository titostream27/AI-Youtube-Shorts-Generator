# Hardening Sprint 2026 — Architecture & Operations Guide

Scope: Podcast Opportunity Miner (content-miner) + AI-Youtube-Shorts-Generator
hardening sprint (v1/v2/v3/v4/v5/v6). Fixes correctness, contract, evaluation, cache,
timeline, and boundary gaps before feature expansion.

## Revision v6 (this sprint) — what changed

- **Renderer state/failure-path (R01..R07).**
  require_transition() checked helper for EVERY active/terminal transition;
  lost CAS raises JobTransitionConflict and stops work. Sync terminal
  persistence return is checked (False never yields a success response).
  Worker exception handler preserves a winning terminal state. quality_check
  is a checked job-level stage; ensure_worker_running() restarts a dead
  worker exactly once and the loop resets the started flag on exit. Health
  gains oldest_queued_age_sec. Production crop/caption paths REQUIRE an
  explicit RenderTimeline (RenderTimelineMissingError); no module-global
  stats fallback anywhere in production.
- **Miner finalization/slicing (M01..M03).**
  finalizeCandidate fully revalidates the FINAL range after start repair
  (duration, ending, contamination, topic boundary) before slicing.
  sliceTranscriptForRange computes wordTimingCoverage and uses honest
  precision word/hybrid/utterance; untimed overlapping speech retained in
  hybrid; no-word fallback labeled 'utterance' not 'cue'.
- **Contract parity (C01..C03).**
  Shared fixtures: 6 new invalid (C-INV-01/02/04/06/08/09). Pydantic
  narrative checks read clip.narrative (finite/in-clip/hook<=payoff); cue
  and event chronological order. Zod language default 'auto' + narrative
  in-clip. RenderResponse strictly typed with RenderArtifactResult;
  QCDetail.status.
- **Visual planner (R07/VIS-01..08).**
  CameraPlanner pure state machine over detections/audio/scene events;
  decision tests: single speaker, no size steal, handoff, miss hold, scene
  reset, split, false face, audio hysteresis.
- **Evaluator (E03).**
  Legacy clipId helpers redirected to canonical AssignmentResult (temporal
  IoU); no production metric depends on clipId equality.
- **CI/docs.**
  Renderer CI pins miner at v6 HEAD; hardening-v6 in blocking set.

## Revision v5 (this sprint) — what changed

- **Phase 1 — renderer lifecycle correctness (R-01..R-07).**
  Sync and async now share one orchestration path (`_register_job_memory` +
  `transition_job` + `_persist_terminal_via_transition`); idempotency hits
  never create phantom memory entries; lost CAS raises JobTransitionConflict
  and never renders; original exception class/message preserved. Terminal
  status persists THROUGH the state machine BEFORE memory update (R-05) —
  memory never claims completed when SQLite fails. `_persist_job` uses named
  SQL parameters (R-03 tuple swap fixed) and raises PersistenceError on
  failure. Queue admission compensates (durable queued -> failed,
  error_stage=queue_admission) before HTTP 503 (R-04). Force lookup DB
  failure fails the request (4.7). Startup reconciliation orphans stale
  foreign-boot rows BEFORE serving (4.6). Worker health uses the real thread
  reference (is_alive), heartbeat, last exception (9.2). Final mode fails
  closed when QC is unavailable (4.5). Trusted caption word timing is
  invalidated after pause trim (R-06).
- **Phase 2 — miner finalization (M-01..M-04).**
  finalizeCandidate() now slices the canonical transcript ONLY AFTER the
  start gate resolves the final start — the final slice always describes
  segment.startSec..endSec (M-01). Debug metadata comes from the final
  result (M-02). Canonical `cue.words` propagate into utterances and drive
  REAL word-level slicing (M-03). Repeated pronouns are NOT antecedents;
  only entity evidence resolves openers (M-04).
- **Phase 3 — contract parity (C-01).**
  Every nested Zod object is strict; every nested Pydantic model forbids
  extras. Clip IDs normalize to non-empty strings before duplicate
  detection (1 == "1"). NaN/Infinity rejected in numeric fields.
  RenderArtifactResult enforces 6.3 invariants (status=ok requires
  video_url; status=error requires error + publishable=false).
- **Phase 4 — timeline/visual (V-01).**
  ReframeResult object (path + timeline + stats + cache_key +
  pipeline_version); caption compositor uses timeline.state_at(ts) per frame
  instead of the final face snapshot. Scenario-specific mocked-detection
  tests added (single speaker, active switch, missed detection, caption
  collision).
- **Phase 5 — evaluation (G-01/G-02).**
  computeAssignmentResult(): positives matched independently; hard-negative
  overlaps reported for every prediction that touches them (a prediction may
  overlap both, both facts reported). Rank-aware metrics iterate labels by
  expected rank, not greedy order.
- **Phase 6 — CI/docs (CI-01).**
  Lint blocks; renderer CI pins the miner contract SHA; requirements
  install fail-fast; visual tests included; test summaries published.

## Revision v4 — what changed

- **Phase A: render job correctness.** Sync jobs now go through the SAME
  durable reservation + in-memory registration as async/retry (transition_job
  can actually advance them; exceptions persist `failed`). V1 async without
  request_id inserts a durable queued row. Multi-clip jobs stay in
  `rendering` until ALL clips render, then enter `quality_check` once —
  per-clip QC no longer mutates job state. Force-rerender does NOT change the
  semantic request hash (execution control is not content) and increments
  attempt monotonically with parent lineage. Double final-encode failure
  fails CLOSED: no lossless intermediate is exposed as a publishable URL.
  `_reserve_job` distinguishes sqlite3.IntegrityError (duplicate race) from
  other DB failures (raise + diagnostics, no worker). Thread-per-job was
  replaced by a bounded `queue.Queue` + one persistent worker; orphan
  detection uses `process_boot_id` ownership + age threshold (fresh current-
  process jobs are never orphaned). Health reports queue depth, worker alive,
  boot id.
- **Phase B: miner candidate correctness.** ONE `finalizeCandidate()` path
  for semantic and repaired candidates: hard start gate (repair-or-reject
  with preceding-context validation), final-slice rescoring (never rough
  salience inheritance), identity/lineage stamping. Paragraph pause flushes
  BEFORE adding the cue; question/transition openers anchored to utterance
  start; word-level slicing clips first/last words to the window.
- **Phase C: contract & captions.** Cues/words are intersected and clamped to
  clip boundaries (zero/negative durations dropped). Language resolves
  explicit -> transcript -> auto (never blind 'en'). Narrative timings are
  per-clip (`narrativeByClipId`), never leaked across clips.
- **Phase D: evaluation.** Temporal assignment uses SEPARATE label/prediction
  namespaces (crossing matches both; one pred cannot match two labels).
  `GoldenLabel.type` = positive | hard_negative | ignore; hard negatives are
  false positives (hardNegativeFPR), never recall; ignores excluded. Top-K
  and rank-aware recall use the same temporal assignment as recall (different
  id but correct window = hit).
- **Phase E: cache/CI.** Timeline sidecars are written on EVERY cache write
  (even when the caller did not request a timeline). GitHub Actions CI added
  for both repos; missing shared contract fixtures fail explicitly.

## Revision v3 — what changed

- **Phase A: reservations.** `_reserve_job` now treats ONLY
  `sqlite3.IntegrityError` as a duplicate-active-request race; any other DB
  failure raises, is recorded in health diagnostics, and never starts a
  worker on an unreserved row.
- **Phase B: cache & timeline.** Cache key salts the render profile
  (camera/caption-safe/tracker/encoder/pipeline), layout mode, editing-event
  hash, and source fingerprint. Timeline frames carry time-indexed state
  (faces, active speaker, camera center, layout, crop); `RenderTimeline`.
  `state_at(t)` returns the nearest-frame state or an explicit
  `no_timeline` — never module globals from a previous render.
- **Phase C: miner finalization.** Candidate id is a content/window sha256
  fingerprint (`c=<fp>`), stable across rough-index shifts. Salience, word
  count and density are recomputed from the FINAL slice
  (`rescoreSegmentFromSlice`), never inherited from the rough candidate.
  Transcript slices expose `timingPrecision`/`sliceApproximate`. Pronoun and
  referential openers resolve against PRECEDING entity context.
- **Phase D: canonical captions.** `_normalize_clips` preserves `words`,
  language, provider, `transcript_version`, `alignment_confidence`; trusted
  canonical word timing (confidence ≥ threshold) skips full Whisper; the STT
  fallback uses the contract language (never hard-codes English); words are
  clamped to the clip; untimed words are marked `timing_source=
  synthetic_hint`.
- **Phase E: shared result contract.** `contracts/render-result-v2.schema.
  json` + valid/invalid fixtures + manifest are a single source of truth for
  the renderer RESULT payload, validated identically by JSON Schema, Zod, and
  Pydantic `RenderResult`. `clip_id` is normalized before duplicate checks;
  `request_id` is salted by `renderProfileVersion`; an end-to-end
  miner→renderer handshake test parses a miner contract with the renderer.
  Pydantic and proves no field loss.
- **Phase F: evaluation.** All boundary-sensitive golden metrics come from ONE
  common temporal assignment (`matchByTemporalIoU`); a 4th golden fixture
  (contamination hard-negative) was added.

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