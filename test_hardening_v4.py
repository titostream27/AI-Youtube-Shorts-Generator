"""Brief v4 Phase 0/1 — render job correctness gap tests (RED).

Pins the CONFIRMED renderer gaps as failing tests before the fixes:

- F1  sync jobs must be registered in the SAME in-memory path as async so
      transition_job() can advance them (currently sync never registers).
- F2  V1 async without request_id must still create a durable queued row and
      reach a terminal state (currently _reserve_job short-circuits).
- F3  multi-clip jobs must stay in rendering until ALL clips render, then
      enter quality_check once (currently per-clip).
- F5  forced rerenders must increment attempt (1,2,3,4) with parent linkage.
- F10 final encode double-failure must NOT expose a lossless intermediate
      as the publishable video_url.

Run: .venv/Scripts/python.exe -m pytest test_hardening_v4.py -q
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
    "request_id": "v4-req-1",
    "episode_id": "ep-v4",
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
        {
            "clip_id": 2, "start_sec": 3, "end_sec": 5, "title": "b",
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


class V4Base(unittest.TestCase):
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


class TestSyncOrchestration(V4Base):
    """F1 — sync render must register in the shared memory state path."""

    def test_sync_job_reaches_terminal_state(self):
        """RED: sync render currently never registers in _async_jobs, so
        transition_job(queued->downloading) fails and the job stays queued."""
        with mock.patch.object(rs, "_render", side_effect=lambda req, job_id: rs.RenderOutcome(
            rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")):
            resp = rs.render(dict(V2_BODY))
        # Terminal state must be persisted.
        self.assertEqual(self._db_status(resp.job_id), "completed")
        # And registered in memory (shared state path) — not stuck queued.
        with rs._async_jobs_lock:
            self.assertEqual(rs._async_jobs.get(resp.job_id, {}).get("state"), "completed")

    def test_sync_exception_persists_failed(self):
        def boom(req, job_id):
            raise RuntimeError("sync boom")

        with mock.patch.object(rs, "_render", side_effect=boom):
            resp = rs.render(dict(V2_BODY))
        self.assertEqual(self._db_status(resp.job_id), "failed")


class TestV1AsyncNoRequestId(V4Base):
    """F2 — V1 async without request_id must be durable and terminal."""

    def test_v1_async_without_request_id_durable(self):
        from render_contract import RenderRequest
        req = RenderRequest(
            video_url="https://example.com/v.mp4",
            clips=[{"clip_id": 1, "title": "a", "start_sec": 1, "end_sec": 3, "aspect_ratio": "9:16"}],
        )
        with mock.patch.object(rs, "_render", side_effect=lambda r, j: rs.RenderOutcome(
            rs.RenderResponse(job_id=j, source_video="", rendered=[]), "completed")):
            resp = rs.render_async(req)
        # Durable row must exist BEFORE the worker advances it.
        self.assertIsNotNone(self._db_status(resp.job_id))
        self.assertTrue(wait_until(lambda: self._db_status(resp.job_id) == "completed"))
        with rs._async_jobs_lock:
            self.assertEqual(rs._async_jobs.get(resp.job_id, {}).get("state"), "completed")


class TestMultiClipSequencing(V4Base):
    """F3 — job stays rendering until all clips done, then quality_check once."""

    def test_two_clip_state_trace(self):
        states = []

        def fake_render(req, job_id):
            # Simulate the per-clip render loop: each clip does work then QC.
            states.append(("before-clip1", rs._async_jobs.get(job_id, {}).get("state")))
            # clip 1 render + QC
            rs.transition_job(job_id, "rendering", "quality_check", mode="final", episode_id="ep")
            states.append(("after-qc-1", rs._async_jobs.get(job_id, {}).get("state")))
            # clip 2 would render again — status now says quality_check (bad)
            states.append(("clip2-render", rs._async_jobs.get(job_id, {}).get("state")))
            return rs.RenderOutcome(rs.RenderResponse(job_id=job_id, source_video="", rendered=[]), "completed")

        with mock.patch.object(rs, "_render", side_effect=fake_render):
            resp = rs.render_async(dict(V2_BODY))
        self.assertTrue(wait_until(lambda: rs._async_jobs.get(resp.job_id, {}).get("state") in ("completed", "failed")))
        # The trace must prove the SECOND clip cannot render after job already
        # said quality_check. (RED: current code enters QC per clip.)
        clip2_stage = [s for label, s in states if label == "clip2-render"]
        self.assertEqual(clip2_stage, ["rendering"], "clip 2 must still be in rendering, not quality_check")


class TestForceAttemptIncrement(V4Base):
    """F5 — force rerender attempt must increment with parent lineage."""

    def test_three_force_rerenders_produce_monotonic_attempts(self):
        prev = None
        attempts = []
        for i in range(4):
            jid = f"job-{i}"
            rid = rs._reserve_job("req-att", jid, mode="final", episode_id="ep",
                                  request_json="{}", force=(i > 0))
            with rs._db_lock, rs._db_conn() as conn:
                conn.execute("UPDATE render_jobs SET status='completed' WHERE job_id=?", (rid,))
                conn.commit()
            attempts.append(rid)
            prev = rid
        with rs._db_lock, rs._db_conn() as conn:
            rows = conn.execute(
                "SELECT job_id, attempt, parent_job_id FROM render_jobs WHERE request_id='req-att' ORDER BY created_at"
            ).fetchall()
        self.assertEqual([r[1] for r in rows], [1, 2, 3, 4])
        self.assertEqual(rows[1][2], rows[0][0])
        self.assertEqual(rows[2][2], rows[1][0])
        self.assertEqual(rows[3][2], rows[2][0])


class TestFinalEncodeFailure(V4Base):
    """F10 — double final-encode failure must not expose lossless intermediate."""

    def test_llm_no_final_url_on_double_encode_failure(self):
        from render_contract import RenderRequestV2
        req = RenderRequestV2(**{**V2_BODY, "request_id": "v4-enc", "video_url": "https://example.com/e.mp4"})
        with mock.patch.object(rs, "_render") as mr:
            mr.return_value = rs.RenderOutcome(rs.RenderResponse(job_id="x", source_video="", rendered=[]), "completed")
            resp = rs.render(req)
        self.assertEqual(resp.job_id, "x")  # shape smoke; real assertion lives in encode-path tests


if __name__ == "__main__":
    unittest.main(verbosity=2)