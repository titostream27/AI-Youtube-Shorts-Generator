# Brief v11 — Mandatory Pre-Implementation Audit

## 1. Repository baselines (current HEAD)

| Repo | HEAD SHA | git status | Branch |
|------|----------|-----------|--------|
| youtube-content-miner | `6e19ce1` | clean | main |
| AI-Youtube-Shorts-Generator | `519b101` | clean | main |

## 2. Production call graph (renderer)

- **POST /api/render** (sync): `render()` -> `_reserve_job(request_id, job_id, force=force_rerender)` -> queue CAS -> `_render()` -> `_persist_terminal_via_transition()`.
- **POST /api/render** (async): `render_async()` -> `get_existing_request_result()` (idempotent) -> `_reserve_job(force=force_rerender)` -> `_enqueue_job()`.
- **POST /api/render/jobs/{job_id}/retry**: `render_job_retry()` -> `_load_job()` -> manual `attempt = parent_attempt + 1` -> `_persist_job()` -> `_enqueue_job()`. **NO reserve_attempt().**
- **GET status / POST cancel**: `render_job_status()` -> `_load_durable_snapshot()`; `render_job_cancel()` -> `transition_job()` then `_load_job()`.

## 3. Schema / indexes (render_jobs)

- Columns: request_id, parent_job_id, attempt, process_boot_id, started_at, finished_at, last_error_stage.
- Indexes:
  - `idx_render_jobs_active_request` UNIQUE partial on request_id WHERE request_id!='' AND status IN (active states). **one-active-attempt guard.**
  - `uq_render_jobs_request_attempt` UNIQUE partial (request_id<>''). **monotonic attempt guard (guarded by stop-condition duplicate precheck).**
  - idx_render_jobs_request_id/status/created_at/parent_job_id.
- NOTE: live `rendered/render_jobs.db` currently contains duplicate `('e2e-verify-idem-1', 1, 2)`; the unique index refuses to build until cleaned (stop condition observed).

## 4. Callers of reserve_attempt() and _reserve_job(force=True)

- `reserve_attempt()`: tests only (test_retry_v10).  **NOT reached by any production route.**
- `_reserve_job(force=True)`: `render()` line 0 (sync) and `render_async()` line 0 (async). Live force path.

## 5. _load_job() exception propagation

- `_load_job()` line 585: catches generic `Exception`, falls back to legacy SELECT, and on a second failure returns `None` (line ~634). **PERSISTENCE/corruption/I-O error masreporzed as job-not-now, violates fail-closed (F11-05).**
- `_find_job_by_request()` already narrows to missing-column only (v10 C07). Status/retry/cancel still treat None as 404.

## 6. Miner repaired look-ahead (F11-06)

- `two-pass.ts:408` repaired branch: `utterances.slice(repEndIdx + 1, repEndIdx + 4)` — fixed three-utterance lookahead. **STILL PRESENT** (must be time-based via followingWithinLookaheadSec).
- Semantic/deterministic/final validation otherwise use followingWithinLookaheadSec (v9). One shared function not yet used by repaired path.

## 7. Hybrid transcript provenance (F11-07)

- `sliceTranscriptForRange()` (utterances.ts:326) already reports timingPrecision/timingCoverage (v10). Downstream release: finalize-candidate. **Partial/hybrid text may still include whole pre-boundary text of a partially-overlaping utterance; excludedOrUncertainText not populated.** Confirm scope in C8.

## 8. Golden matcher objective (F11-09)

- `matchByTemporalIoU()` implemented as min-cost max-flow lexicographic (max-card, then max total IoU) — v10 C11. **ALREADY FIXED at v10; EV11 tests to prove EV11-01..05.**

## 9. Test counts / evidence

- miner: `npx vitest run` 237 passed; `npx tsc --noEmit` clean.
- renderer: `pytest -q` 185 passed; 1 transient teardown PermissionError (win temp lock, not code regression).
- EMLEe media evidence: `evidence_out/*_final.mp4` are v8 local fixtures, NOT real podcast acceptance; no ~10 real titles yet.

## STOP-CONDITION NOTE

State does NOT match brief assumptions in: matcher already lexicographic (v10). No blind reimplementation required. Remaining findings F11-01..05, F11-07, F11-08, F11-10 still confirmed/active.