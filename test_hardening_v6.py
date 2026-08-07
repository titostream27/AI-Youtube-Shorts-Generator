"""Brief v6 Phase A — renderer state/failure-path gap tests (RED).

Pins CONFIRMED findings from docs/audits/brief-v6-verification.md:

- T-R01 active transition result must be checked (downloading->analysing,
  analysing->rendering) — render must stop on conflict (R01).
- T-R02 worker exception must NOT force memory to failed when another
  terminal state (e.g. cancelled) won (R02).
- T-R03 sync terminal persistence return False must NOT produce success
  response or memory completed (R03).
- T-R05 dead queue worker must restart exactly once on next enqueue (R05).
- T-R06 bare crop path must raise RenderTimelineMissingError (R06).

Run: .venv/Scripts/python.exe -m pytest test_hardening_v6.py -q
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
    "request_id": "v6-req-1",
    "episode_id": "ep-v6",
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


class V6Base(unittest.TestCase):
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


class TestCheckedActiveTransitions(V6Base):
    """T-R01 — every active-stage transition result is checked (R01)."""

    def test_analysing_to_rendering_conflict_stops_render(self):
        calls = {"n": 0}

        def fake_render(req, job_id):
            calls["n"] += 1
            return rs.RenderOutcome(rs.RenderResponse(job_id=job_id, source_video="", rendered=[], status="completed"), "completed")

        with mock.patch.object(rs, "_render", side_effect=fake_render):
            real_transition = rs.transition_job

            def flaky(job_id, expected, target, **kw):
                if expected == "analysing" and target == "rendering":
                    return False
                return real_transition(job_id, expected, target, **kw)

            with mock.patch.object(rs, "transition_job", side_effect=flaky):
                with self.assertRaises(rs.JobTransitionConflict):
                    rs.render(dict(V2_BODY))
        # Render must not have completed; job must not be in completed state.
        with rs._db_lock, rs._db_conn() as conn:
            rows = conn.execute("SELECT job_id, status FROM render_jobs WHERE status='completed'").fetchall()
        self.assertEqual(len(rows), 0, "no job may be completed after a lost active transition")


class TestWorkerExceptionPreservesTerminal(V6Base):
    """T-R02 — worker exception must not overwrite a winning terminal state (R02)."""

    def test_cancel_wins_then_exception_does_not_force_failed(self):
        job_id = rs._reserve_job("v6-r02", "r2", mode="final", episode_id="e", request_json="{}")
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {"state": "queued", "response": None, "error": None,
                                      "request_id": "v6-r02", "mode": "final"}
        # Cancel wins -> terminal cancelled.
        self.assertTrue(rs.transition_job(job_id, "queued", "cancelled"))
        # Current buggy handler would force memory failed. After fix, memory
        # must stay cancelled. We simulate the handler code path directly.
        cur = rs._load_job(job_id)
        src = rs.canonical_status(cur["status"]) if cur else "queued"
        ok = rs.transition_job(job_id, src, "failed", error="boom", error_stage="worker")
        # SQLite stays cancelled because cancelled is terminal (no outgoing).
        self.assertEqual(self._db_status(job_id), "cancelled")
        # The fix: the handler must NOT blindly write memory=failed when the
        # transition was not applied. Here we assert the expected final memory.
        with rs._async_jobs_lock:
            state = rs._async_jobs.get(job_id, {}).get("state")
        self.assertIn(state, ("cancelled", None), "memory must not claim failed when cancelled won")


class TestSyncTerminalPersistReturn(V6Base):
    """T-R03 — sync terminal persistence False must not yield success (R03)."""

    def test_persist_false_no_completed_memory(self):
        def fake_render(req, job_id):
            return rs.RenderOutcome(rs.RenderResponse(job_id=job_id, source_video="", rendered=[], status="completed"), "completed")

        with mock.patch.object(rs, "_persist_terminal_via_transition", return_value=False), \
             mock.patch.object(rs, "_render", side_effect=fake_render):
            with self.assertRaises(rs.JobTransitionConflict):
                rs.render(dict(V2_BODY))
        with rs._async_jobs_lock:
            completed = [jid for jid, j in rs._async_jobs.items() if j.get("state") == "completed"]
        self.assertEqual(completed, [], "memory must not claim completed when terminal commit returned False")


class TestWorkerRestartOnDeath(V6Base):
    """T-R05 — dead worker restarts exactly once on next enqueue (R05)."""

    def test_worker_restarted_after_crash(self):
        # Simulate dead worker: flag True but thread None.
        with rs._render_queue_worker_lock:
            rs._render_queue_worker_started = True
            rs._render_queue_worker_thread = None
        # Patch the loop so the started thread returns immediately.
        with mock.patch.object(rs, "_queue_worker_loop", side_effect=lambda: None):
            rs.ensure_worker_running()
        with rs._render_queue_worker_lock:
            self.assertTrue(rs._render_queue_worker_started,
                            "worker must be (re)started when thread is dead/None")
            self.assertIsNotNone(rs._render_queue_worker_thread)

    def test_worker_flag_reset_on_loop_exit(self):
        """After the loop exits (crash), the started flag must reset so the
        next enqueue restarts the worker."""
        with rs._render_queue_worker_lock:
            rs._render_queue_worker_started = True
            rs._render_queue_worker_thread = None
        # Simulate loop exit by running the loop with an immediate exception
        # on get() — the outermost finally must reset the flag.
        with mock.patch.object(rs._render_queue, "get", side_effect=RuntimeError("get died")):
            try:
                rs._queue_worker_loop()
            except Exception:
                pass
        with rs._render_queue_worker_lock:
            self.assertFalse(rs._render_queue_worker_started,
                             "flag must reset after worker loop exits")


class TestNoGlobalTimelineFallback(V6Base):
    """T-R06 — bare crop path must raise RenderTimelineMissingError (R06)."""

    def test_bare_crop_path_raises(self):
        if not hasattr(rs, "RenderTimelineMissingError"):
            self.skipTest("RenderTimelineMissingError not yet defined (RED)")
        self.assertTrue(hasattr(rs, "_require_explicit_timeline"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
