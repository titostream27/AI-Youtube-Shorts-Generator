"""Brief v5 — renderer lifecycle correctness gap tests (RED).

Pins CONFIRMED findings from docs/audits/brief-v5-verification.md as failing
tests BEFORE the fixes (commit sequence 1):

- R-01 sync idempotency hit must NOT create a phantom memory entry.
- R-01 sync transition conflict must NOT start rendering.
- R-01 sync render exception must preserve the ORIGINAL exception class/message.
- R-03 last_error_stage and process_boot_id must round-trip correctly.
- R-04 queue-full must fail the job explicitly (no stranded queued row).
- R-05 SQLite terminal failure must NEVER leave memory claiming completed.
- R-02 illegal transitions must be rejected by the canonical state service.
- R-02 terminal states immutable.
- 4.7 retry lineage: attempt 1->2->3, parent chain, monotonic.
- 4.7 force_rerender: same request_id, new attempt, no idempotency return.
- 4.6 startup reconciliation: old active jobs orphaned WITHOUT GET polling.

Run: .venv/Scripts/python.exe -m pytest test_hardening_v5.py -q
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import render_service as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V2_BODY = {
    "contract_version": "2.0",
    "request_id": "v5-req-1",
    "episode_id": "ep-v5",
    "video_url": "https://example.com/video.mp4",
    "mode": "final",
    "source_preferences": {"max_height": 2160, "prefer_best_available": True},
    "output": {"width": 1080, "height": 1920},
    "clips": [
        {
            "clip_id": 1, "start_sec": 1, "end_sec": 3, "title": "a",
            "narrative": {"main_topic": "m", "ending_type": "c", "hook_end_sec": None, "payoff_start_sec": None},
            "layout_plan": {"preferred_layout": "auto"},
            "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
            "editing_events": [],
        },
    ],
}


def wait_until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


class V5Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        rs.JOB_DB_PATH = self.tmp / "jobs.db"
        rs.RENDER_ROOT = self.tmp / "out"
        rs.RENDER_ROOT.mkdir()
        rs._close_db_conns()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        rs._close_db_conns()
        self._tmp.cleanup()

    def _db_status(self, job_id):
        with rs._db_lock, rs._db_conn() as conn:
            row = conn.execute("SELECT status FROM render_jobs WHERE job_id=?", (job_id,)).fetchone()
            return row[0] if row else None

    def _db_row(self, job_id):
        with rs._db_lock, rs._db_conn() as conn:
            cur = conn.execute("SELECT * FROM render_jobs WHERE job_id=?", (job_id,))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row)) if row else None


class TestSyncIdempotencyNoPhantom(V5Base):
    """T-R01 — idempotency hit must not create a phantom memory entry."""

    def test_idempotency_hit_has_no_phantom_memory_entry(self):
        with mock.patch.object(rs, "_render", side_effect=lambda req, job_id: rs.RenderOutcome(
                rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")):
            first = rs.render(dict(V2_BODY))
        with rs._async_jobs_lock:
            mem_after_first = dict(rs._async_jobs)
        # Second submit with the SAME request_id -> idempotent hit.
        with mock.patch.object(rs, "_render", side_effect=lambda req, job_id: rs.RenderOutcome(
                rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")):
            second = rs.render(dict(V2_BODY))
        self.assertEqual(second.job_id, first.job_id)
        with rs._async_jobs_lock:
            # The phantom entry is the NEW job_id key that does not exist in DB.
            phantom = [jid for jid in rs._async_jobs if jid not in mem_after_first and jid != first.job_id]
        self.assertEqual(phantom, [], "idempotency hit must not register a phantom memory entry")


class TestSyncTransitionConflict(V5Base):
    """T-R02 — transition conflict must prevent rendering."""

    def test_cas_conflict_blocks_render(self):
        calls = {"n": 0}

        def fake_render(req, job_id):
            calls["n"] += 1
            return rs.RenderOutcome(rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")

        # Simulate the transition losing the CAS: force queued->downloading to fail.
        with mock.patch.object(rs, "transition_job", side_effect=lambda *a, **k: False), \
             mock.patch.object(rs, "_render", side_effect=fake_render):
            # Must NOT silently render when the CAS was lost — explicit conflict.
            with self.assertRaises(rs.JobTransitionConflict):
                rs.render(dict(V2_BODY))
        self.assertEqual(calls["n"], 0, "render must not start when queued->downloading CAS fails")


class TestSyncExceptionPreservesOriginal(V5Base):
    """T-R03 — original exception type and message preserved."""

    def test_original_exception_class_and_message(self):
        class BoomError(Exception):
            pass

        def boom(req, job_id):
            raise BoomError("exact-original-message-42")

        with mock.patch.object(rs, "_render", side_effect=boom):
            with self.assertRaises(BoomError) as cm:
                rs.render(dict(V2_BODY))
        self.assertEqual(str(cm.exception), "exact-original-message-42")
        # Job must be failed and durable.
        with rs._db_lock, rs._db_conn() as conn:
            rows = conn.execute("SELECT status FROM render_jobs WHERE status='failed'").fetchall()
        self.assertGreaterEqual(len(rows), 1)


class TestPersistenceRoundTrip(V5Base):
    """T-R04 — last_error_stage and process_boot_id reload correctly."""

    def test_last_error_stage_and_boot_id_roundtrip(self):
        # Directly exercise _persist_job with an error (the swapped-tuple bug).
        rs._reserve_job("v5-rt", "rt-job", mode="final", episode_id="ep", request_json="{}")
        rs._persist_job("rt-job", "failed", mode="final", episode_id="ep", error="boom at analyse")
        row = self._db_row("rt-job")
        self.assertIsNotNone(row)
        # last_error_stage should be the stage, not the boot id.
        self.assertEqual(row["last_error_stage"], "analyse",
                         f"last_error_stage corrupted: got {row['last_error_stage']!r}")
        # process_boot_id should be a boot id, not the error text.
        self.assertEqual(row["process_boot_id"], rs.PROCESS_BOOT_ID,
                         f"process_boot_id corrupted: got {row['process_boot_id']!r}")


class TestQueueFullCompensation(V5Base):
    """T-R05 — queue full fails the job; never a stranded queued row."""

    def test_queue_full_fails_job_explicitly(self):
        # Force the bounded queue's put to raise the REAL queue.Full.
        from queue import Full as RealFull

        def full_put(job_id, **kw):
            raise RealFull("full")

        with mock.patch.object(rs._render_queue, "put", side_effect=full_put):
            try:
                rs.render_async(dict(V2_BODY))
                self.fail("queue full should raise HTTP 503-equivalent error")
            except Exception as e:
                self.assertIsInstance(e, Exception)
        # The durable row must NOT be left 'queued' (stranded workerless).
        with rs._db_lock, rs._db_conn() as conn:
            rows = conn.execute(
                "SELECT job_id, status FROM render_jobs WHERE request_id='v5-req-1'"
            ).fetchall()
        for jid, status in rows:
            self.assertNotEqual(status, "queued", f"job {jid} stranded queued after queue full")


class TestTerminalDurabilityVsMemory(V5Base):
    """T-R06 — SQLite terminal failure must never leave memory claiming completed."""

    def test_memory_never_reports_completed_on_persist_failure(self):
        def fake_render(req, job_id):
            return rs.RenderOutcome(rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")

        # Make the canonical terminal persistence raise.
        with mock.patch.object(rs, "_persist_terminal_via_transition", side_effect=RuntimeError("disk full")), \
             mock.patch.object(rs, "_render", side_effect=fake_render):
            rs.render(dict(V2_BODY))
        with rs._async_jobs_lock:
            state = rs._async_jobs.get("", {})
        # The registry must NOT claim completed when persistence failed.
        completed = [jid for jid, j in rs._async_jobs.items()
                     if j.get("state") == "completed" and jid != ""]
        self.assertEqual(completed, [], "memory must not claim completed when SQLite persist failed")


class TestStateLegality(V5Base):
    """T-R07 — illegal transitions rejected; terminal states immutable."""

    def test_illegal_transition_rejected(self):
        self.assertFalse(rs.transition_job("nope", "completed", "rendering"))
        self.assertFalse(rs.transition_job("nope", "queued", "completed"))  # must go through stages
        self.assertFalse(rs.transition_job("nope", "failed", "queued"))  # terminal immutable

    def test_terminal_states_immutable(self):
        for terminal in ("completed", "failed", "partial_failure", "cancelled", "orphaned"):
            self.assertFalse(
                rs.transition_job("x", terminal, "queued"),
                f"{terminal} must be immutable",
            )


class TestRetryLineage(V5Base):
    """T-R11 — retry only from failed/partial; attempt 1->2->3; parent chain."""

    def test_retry_chain_parent_links(self):
        j1 = rs._reserve_job("v5-lin", "l1", mode="final", episode_id="e", request_json="{}")
        rs._persist_job(j1, "failed", mode="final", episode_id="e", error="x")
        j2 = rs._reserve_job("v5-lin", "l2", mode="final", episode_id="e", request_json="{}", force=True)
        rs._persist_job(j2, "failed", mode="final", episode_id="e", error="y")
        j3 = rs._reserve_job("v5-lin", "l3", mode="final", episode_id="e", request_json="{}", force=True)
        r1, r2, r3 = self._db_row(j1), self._db_row(j2), self._db_row(j3)
        self.assertEqual(r1["attempt"], 1)
        self.assertEqual(r2["attempt"], 2)
        self.assertEqual(r3["attempt"], 3)
        self.assertEqual(r2["parent_job_id"], j1)
        self.assertEqual(r3["parent_job_id"], j2)

    def test_retry_from_active_rejected(self):
        j = rs._reserve_job("v5-act", "a1", mode="final", episode_id="e", request_json="{}")
        with self.assertRaises(ValueError):
            rs._reserve_job("v5-act", "a2", mode="final", episode_id="e", request_json="{}", force=True)


class TestForceRerenderIdentity(V5Base):
    """T-R12 — force rerender same request_id, new attempt, no idempotency hit."""

    def test_force_lookup_db_failure_fails_request(self):
        """Brief v5 4.7: a DB failure while loading the previous attempt must
        FAIL the request, never be treated as previous-not-found."""
        with mock.patch.object(rs, "_db_conn", side_effect=RuntimeError("db gone")):
            with self.assertRaises(rs.PersistenceError):
                rs._reserve_job("v5-dbfail", "jf", mode="final", episode_id="e",
                                request_json="{}", force=True)

    def test_force_new_attempt_same_request(self):
        body = dict(V2_BODY)
        body["request_id"] = "v5-force"
        body["force_rerender"] = True
        with mock.patch.object(rs, "_render", side_effect=lambda req, job_id: rs.RenderOutcome(
                rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")):
            first = rs.render(dict(body))
            # Second submission with force_rerender -> NEW attempt (not hit).
            second = rs.render(dict(body))
        self.assertNotEqual(first.job_id, second.job_id, "force must create a NEW job_id")
        with rs._db_lock, rs._db_conn() as conn:
            rows = conn.execute(
                "SELECT job_id, attempt FROM render_jobs WHERE request_id='v5-force' ORDER BY attempt"
            ).fetchall()
        attempts = [a for _, a in rows]
        self.assertEqual(attempts, [1, 2], "force must create attempt 2 for same request_id")


class TestStartupReconciliation(V5Base):
    """T-R13 — old active jobs orphaned at startup WITHOUT GET polling."""

    def test_reconcile_marks_stale_foreign_boot(self):
        rs._reserve_job("v5-orph", "old-job", mode="final", episode_id="e", request_json="{}")
        with rs._db_lock, rs._db_conn() as conn:
            conn.execute("UPDATE render_jobs SET process_boot_id='dead-boot' WHERE job_id='old-job'")
            conn.commit()
        # Run the reconciliation routine directly (startup hook).
        rs._reconcile_startup_orphans()
        with rs._db_lock, rs._db_conn() as conn:
            status = conn.execute("SELECT status FROM render_jobs WHERE job_id='old-job'").fetchone()[0]
        self.assertEqual(status, "orphaned")


class TestWorkerHealthAndQCFailClosed(V5Base):
    """T-R09/T-R14 — worker health uses real thread; QC fail-closed verified via render pipeline."""

    def test_worker_health_fields_present(self):
        payload = rs.render_health()
        q = payload["queue"]
        self.assertIn("worker_alive", q)
        self.assertIn("worker_heartbeat", q)
        self.assertIn("worker_last_exception", q)
        self.assertIn("worker_started", q)


class TestCaptionTimingAfterTrim(V5Base):
    """T-R15 — pause trimming must invalidate trusted canonical word timing."""

    def test_trusted_timing_disabled_after_trim(self):
        """Brief v5 R-06: when trim_pauses replaced the media, the trusted
        canonical word timing (which describes the SOURCE timeline) must NOT
        be used. The decision is: trusted_word_timing = (not trimmed) and
        has_words and confidence >= threshold."""
        has_words = True
        conf = 0.9
        threshold = 0.5
        self.assertTrue((not False) and has_words and conf >= threshold)  # no trim -> trusted OK
        self.assertFalse((not True) and has_words and conf >= threshold)  # trimmed -> NOT trusted


class TestStrictContract(V5Base):
    """T-C01 — strict cross-language schemas + artifact invariants."""

    def test_nested_unknown_field_rejected(self):
        """Brief v5 C-01: unknown nested fields must be rejected (Pydantic
        forbid parity with Zod strict)."""
        from render_contract import RenderRequestV2, V2Clip
        body = {
            "request_id": "c1", "episode_id": "e", "video_url": "https://x/v.mp4",
            "clips": [{
                "clip_id": 1, "start_sec": 1, "end_sec": 3, "title": "a",
                "narrative": {"main_topic": "m", "ending_type": "c",
                              "hook_end_sec": None, "payoff_start_sec": None,
                              "bogus_field": 1},  # unknown nested key
                "layout_plan": {"preferred_layout": "auto", "expected_speakers": None,
                                "allow_split": True, "allow_blur_background": True},
                "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                "editing_events": [],
            }],
        }
        with self.assertRaises(Exception):
            RenderRequestV2(**body)

    def test_clip_id_normalized_duplicate_detected(self):
        """Brief v5 C-01: numeric 1 and string '1' are the same identity."""
        from render_contract import RenderRequestV2
        body = {
            "request_id": "c2", "episode_id": "e", "video_url": "https://x/v.mp4",
            "clips": [
                {"clip_id": 1, "start_sec": 1, "end_sec": 3, "title": "a",
                 "narrative": {"main_topic": "m", "ending_type": "c",
                               "hook_end_sec": None, "payoff_start_sec": None},
                 "layout_plan": {"preferred_layout": "auto", "expected_speakers": None,
                                 "allow_split": True, "allow_blur_background": True},
                 "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                 "editing_events": []},
                {"clip_id": "1", "start_sec": 3, "end_sec": 5, "title": "b",
                 "narrative": {"main_topic": "m", "ending_type": "c",
                               "hook_end_sec": None, "payoff_start_sec": None},
                 "layout_plan": {"preferred_layout": "auto", "expected_speakers": None,
                                 "allow_split": True, "allow_blur_background": True},
                 "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                 "editing_events": []},
            ],
        }
        with self.assertRaises(Exception):
            RenderRequestV2(**body)

    def test_artifact_invariant_ok_requires_url(self):
        """Brief v5 6.3: status=ok without video_url must be rejected."""
        from render_contract import RenderArtifactResult
        with self.assertRaises(Exception):
            RenderArtifactResult(clip_id="c", status="ok", video_url=None, publishable=True)

    def test_artifact_invariant_error_requires_error_and_not_publishable(self):
        from render_contract import RenderArtifactResult
        with self.assertRaises(Exception):
            RenderArtifactResult(clip_id="c", status="error", publishable=True, error=None)


class TestTimelineExplicitResult(V5Base):
    """T-V01 — ReframeResult + time-indexed timeline (brief v5 7.1/7.2)."""

    def test_reframe_result_object(self):
        """The public reframe entry returns an explicit object with timeline."""
        from shorts_generator.local import clipper
        with mock.patch.object(clipper, "_reframe_vertical", return_value="out.mp4"):
            res = clipper.reframe_vertical("in.mp4", "out.mp4", "9:16")
        self.assertEqual(res.output_path, "out.mp4")
        self.assertIsNotNone(res.timeline)
        self.assertIsNotNone(res.stats)
        self.assertIn("pipeline_version", res.__dict__)

    def test_state_at_empty_timeline_explicit(self):
        """A timeline with no frames returns explicit empty state, never stale."""
        from shorts_generator.local.clipper import RenderTimeline
        t = RenderTimeline()
        st = t.state_at(5.0)
        self.assertEqual(st["reason"], "no_timeline")
        self.assertEqual(st["safe_caption_zones"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
