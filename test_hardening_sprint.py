"""Hardening Sprint (6 Aug 2026) regression tests — Phase A race & state rules.

These tests pin the P0.1/P1.R1/P1.R2 contracts described in the hardening
sprint brief BEFORE the fixes land, so they are expected to FAIL against the
current main. The fixes then make them pass:

  P0.1  Atomic compare-and-swap transitions: exactly one queued->target
        transition wins; worker renders only after queued->downloading
        succeeds; cancel succeeds only via queued->cancelled.
  P1.R1 Validated state machine: illegal transitions rejected.
  P1.R2 Retry only from failed / partial_failure.

Run with:
    .venv/Scripts/python.exe -m pytest test_hardening_sprint.py -q
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
    "request_id": "req-harden-1",
    "episode_id": "ep-harden",
    "video_url": "https://www.youtube.com/watch?v=hardening001",
    "mode": "final",
    "clips": [
        {
            "clip_id": 1,
            "start_sec": 1.0,
            "end_sec": 5.0,
            "title": "t",
            "narrative": {"main_topic": "m", "ending_type": "c"},
            "caption_plan": {"cues": []},
        }
    ],
}


def wait_until(pred, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


class HardeningTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "jobs.db"
        self.out_root = self.tmp_path / "out"
        self.out_root.mkdir()
        self._patchers = [
            mock.patch.object(rs, "JOB_DB_PATH", self.db_path),
            mock.patch.object(rs, "RENDER_ROOT", self.out_root),
        ]
        for p in self._patchers:
            p.start()
            self.addCleanup(p.stop)
        rs._close_db_conns()
        with rs._async_jobs_lock:
            rs._async_jobs.clear()

    def tearDown(self):
        rs._close_db_conns()
        self._tmp.cleanup()

    def seed_job(self, job_id, state="queued"):
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {
                "state": state, "response": None, "error": None,
                "request_id": f"req-{job_id}", "mode": "final",
            }
        rs._persist_job(job_id, state, mode="final", episode_id="ep-harden")


class TestAtomicTransition(HardeningTestBase):
    """P0.1 — compare-and-swap: exactly one queued transition wins."""

    def test_concurrent_transitions_have_single_winner(self):
        """T01 — a worker winning queued->downloading and a cancel
        winning queued->cancelled can never BOTH succeed."""
        job_id = "race-1"
        self.seed_job(job_id, "queued")

        results = []
        barrier = threading.Barrier(2)

        def worker_cas():
            barrier.wait()
            results.append(rs.transition_job(job_id, "queued", "downloading"))

        def cancel_cas():
            barrier.wait()
            results.append(rs.transition_job(job_id, "queued", "cancelled"))

        t1 = threading.Thread(target=worker_cas)
        t2 = threading.Thread(target=cancel_cas)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        winners = [r for r in results if r is True]
        self.assertEqual(len(winners), 1, f"exactly one CAS wins, got {results}")
        final = rs._load_job(job_id)
        self.assertIsNotNone(final)
        self.assertIn(
            final["status"],
            {"downloading", "cancelled"},
            f"terminal state must be one of the winners, got {final['status']}",
        )
        # Memory and SQLite must agree (T02).
        with rs._async_jobs_lock:
            mem_state = rs._async_jobs[job_id]["state"]
        self.assertEqual(mem_state, final["status"])

    def test_transition_returns_false_for_wrong_expected(self):
        """T03 — out-of-order transition is rejected, not silently applied."""
        job_id = "order-1"
        self.seed_job(job_id, "downloading")
        ok = rs.transition_job(job_id, "queued", "analysing")
        self.assertFalse(ok)
        final = rs._load_job(job_id)
        self.assertEqual(final["status"], "downloading")

    def test_terminal_state_is_immutable(self):
        """T03 — no transition out of a terminal state."""
        for terminal in ("completed", "failed", "partial_failure", "cancelled", "orphaned"):
            job_id = f"term-{terminal}"
            self.seed_job(job_id, terminal)
            ok = rs.transition_job(job_id, terminal, "rendering")
            self.assertFalse(ok, f"{terminal} must be immutable")
            final = rs._load_job(job_id)
            self.assertEqual(final["status"], terminal)


class TestRetrySourceValidation(HardeningTestBase):
    """P1.R2 — only failed and partial_failure may be retried."""

    def test_retry_rejects_completed(self):
        self.seed_job("retry-completed", "completed")
        with self.assertRaises(Exception):
            rs.render_job_retry("retry-completed")

    def test_retry_rejects_queued(self):
        self.seed_job("retry-queued", "queued")
        with self.assertRaises(Exception):
            rs.render_job_retry("retry-queued")

    def test_retry_rejects_cancelled(self):
        self.seed_job("retry-cancelled", "cancelled")
        with self.assertRaises(Exception):
            rs.render_job_retry("retry-cancelled")

    def test_retry_accepts_failed_and_partial_failure(self):
        for state in ("failed", "partial_failure"):
            job_id = f"retry-ok-{state}"
            self.seed_job(job_id, state)
            # Unique request_id per retry so idempotency DB doesn't collide.
            req = dict(V2_BODY, request_id=f"req-retry-{state}")
            with mock.patch.object(rs, "_load_job_request", return_value=req):
                resp = rs.render_job_retry(job_id)
            self.assertEqual(resp["original_job_id"], job_id)
            self.assertEqual(resp["state"], "queued")
            # New attempt carries parent lineage.
            new_id = resp["job_id"]
            stored = rs._load_job(new_id)
            self.assertEqual(stored["parent_job_id"], job_id)
            self.assertGreaterEqual(stored["attempt"], 2)


class TestPartialFailureParity(HardeningTestBase):
    """P1.R3 — partial_failure exposes successful artifacts identically."""

    def test_partial_failure_memory_and_sqlite_both_expose_artifacts(self):
        job_id = "pf-parity"
        # Brief v6 6.3: RenderResponse is strictly typed — artifacts are
        # RenderArtifactResult objects, not raw dicts.
        rendered = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/out/short_01.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="error", error={"message": "encode failed"}, publishable=False, qc_status="failed"),
        ]
        response = rs.RenderResponse(job_id=job_id, source_video="src.mp4", rendered=rendered, status="completed")
        rs._persist_job(
            job_id, "partial_failure", mode="final", episode_id="ep-harden",
            response=response.model_dump_json(),
        )
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {"state": "partial_failure", "response": response, "error": None}

        stored = rs._load_job(job_id)
        mem = None
        with rs._async_jobs_lock:
            mem = rs._async_jobs[job_id]
        self.assertEqual(stored["status"], "partial_failure")
        self.assertEqual(mem["state"], "partial_failure")
        # Successful artifact visible from BOTH stores.
        mem_rendered = mem["response"].rendered if mem.get("response") else []
        stored_rendered = (stored.get("response") or {}).get("rendered", [])
        ok_artifacts_mem = [r for r in mem_rendered if r.status == "ok"]
        ok_artifacts_db = [r for r in stored_rendered if r.get("status") == "ok"]
        self.assertEqual(len(ok_artifacts_mem), 1)
        self.assertEqual(len(ok_artifacts_db), 1)
        self.assertEqual(ok_artifacts_db[0]["video_url"], "/out/short_01.mp4")

    def test_status_endpoint_exposes_partial_failure_artifacts_from_memory(self):
        """P1.R3 — /api/render/status/{job_id} must expose successful artifacts
        for a partial_failure kept in memory, identically to the persisted path."""
        job_id = "pf-http"
        rendered = [
            rs.RenderArtifactResult(clip_id="1", status="ok", video_url="/out/short_01.mp4", publishable=True, qc_status="passed"),
            rs.RenderArtifactResult(clip_id="2", status="error", error={"message": "encode failed"}, publishable=False, qc_status="failed"),
        ]
        response = rs.RenderResponse(job_id=job_id, source_video="src.mp4", rendered=rendered, status="completed")
        with rs._async_jobs_lock:
            rs._async_jobs[job_id] = {
                "state": "partial_failure", "response": response, "error": None,
                "request_id": f"req-{job_id}", "mode": "final",
            }
        payload = rs.render_job_status(job_id)
        self.assertEqual(payload["state"], "partial_failure")
        self.assertIn("rendered", payload)
        self.assertEqual(len(payload["rendered"]), 2)
        ok = [r for r in payload["rendered"] if r.status == "ok"]
        self.assertEqual(len(ok), 1)
        self.assertEqual(ok[0].video_url, "/out/short_01.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
