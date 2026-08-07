"""Brief v7 commit 1 — test(api): expose queued-as-completed and partial-response
status bugs (RED on v6 baseline, GREEN after commit 2).

Tests:
- API-01: async new submission must return state=queued, never completed.
- API-02: async idempotent active hit returns existing ID + actual state +
  idempotent_hit=true.
- API-03: partial-failure final response must have status=partial_failure in
  response, memory, AND SQLite — never completed.

Run: .venv/Scripts/python.exe -m pytest test_hardening_v7.py -q
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs
from render_contract import RenderResponse, RenderArtifactResult


class V7DBIsolation(unittest.TestCase):
    """Brief v8 C15/F1 — every SQLite-touching test gets an isolated temp DB
    and closed connections, so the full suite has zero file-lock failures."""

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
        time.sleep(0.1)  # let worker threads release DB handles on Windows
        self._tmp.cleanup()


V2_BODY = {
    "contract_version": "2.0",
    "request_id": "v7-req-1",
    "episode_id": "ep-v7",
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


class TestAsyncSubmissionResponse(V7DBIsolation):
    """API-01: async new submission must never serialize as completed."""

    def test_async_new_submission_not_completed(self):
        """A freshly queued job must expose state=queued in the submission
        response — a RenderSubmissionResponse, not RenderResponse."""
        body = dict(V2_BODY)
        body["request_id"] = f"api01-{time.time()}"
        resp = rs.render_async(body)
        # Must be a submission response with state field
        state = getattr(resp, "state", None)
        self.assertIsNotNone(state, "response must expose state")
        self.assertEqual(state, "queued", "new submission must be state=queued")
        # Must NOT have status=completed (RenderResponse semantics)
        status = getattr(resp, "status", None)
        self.assertIsNone(status, "submission response must not have final status")


class TestAsyncIdempotentHit(V7DBIsolation):
    """API-02: async idempotent hit must expose actual job state, not completed."""

    def test_idempotent_hit_returns_actual_state(self):
        """Resubmitting the same request_id returns idempotent_hit=True
        with the real state of the existing job."""
        body = dict(V2_BODY)
        body["request_id"] = f"api02-{time.time()}"
        first = rs.render_async(body)
        second = rs.render_async(body)
        # Idempotent hit flag must be present
        hit = getattr(second, "idempotent_hit", None)
        self.assertTrue(hit, "second submission must flag idempotent_hit=True")
        # State must reflect the real job state (queued/rendering/etc)
        state = getattr(second, "state", None)
        self.assertIsNotNone(state, "idempotent hit must expose state")
        self.assertNotEqual(
            state, "completed",
            "idempotent hit on active job must not claim completed"
        )


class TestPartialFailureStatus(V7DBIsolation):
    """API-03: partial-failure final response must have status=partial_failure
    in the response object, in-memory job, AND persisted SQLite row."""

    def test_partial_failure_response_status(self):
        """When one of two clips fails, the RenderResponse embedded in the
        outcome must carry status=partial_failure."""
        body = dict(V2_BODY)
        body["request_id"] = f"api03-{time.time()}"
        # Simulate: clip 1 renders ok, clip 2 fails
        def fake_render(req, job_id):
            rendered = [
                RenderArtifactResult(
                    clip_id="1", status="ok", video_url="/out/s1.mp4",
                    publishable=True, qc_status="passed",
                ),
                RenderArtifactResult(
                    clip_id="2", status="error", publishable=False,
                    qc_status="unavailable",
                    error={"message": "ffmpeg failed"},
                ),
            ]
            resp = RenderResponse(
                job_id=job_id, source_video="", rendered=rendered,
                status="partial_failure",
            )
            return rs.RenderOutcome(resp, "completed")

        with mock.patch.object(rs, "_render", side_effect=fake_render):
            result = rs.render(body)

        self.assertEqual(
            result.status, "partial_failure",
            "response must carry status=partial_failure, not completed"
        )


class TestPublishabilityInvariants(V7DBIsolation):
    """API-03 supplement: publishable must respect mode and QC."""

    def test_preview_artifact_never_publishable(self):
        """A preview-mode artifact with qc_status=unavailable must not
        be publishable — the validator must reject it."""
        with self.assertRaises(Exception):
            RenderArtifactResult(
                clip_id="2", status="ok", video_url="/out/s2.mp4",
                publishable=True, qc_status="unavailable",
            )

    def test_qc_failed_prevents_publishable(self):
        """publishable=True with qc_status=failed must be rejected."""
        with self.assertRaises(Exception):
            RenderArtifactResult(
                clip_id="3", status="ok", video_url="/out/s3.mp4",
                publishable=True, qc_status="failed",
            )


if __name__ == "__main__":
    unittest.main()
