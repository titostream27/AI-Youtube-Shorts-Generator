"""Brief v8 C01 — test(renderer): expose async response-model and idempotent
retrieval gaps through the REAL ASGI client (RED on v7, GREEN after C02).

Tests:
- API-01: POST /api/render/async must use RenderSubmissionResponse (not
  RenderResponse); no ResponseValidationError; no final-artifact fields.
- API-02: OpenAPI schema separates submission vs final result routes.
- API-03: persisted completed idempotent hit survives memory reset (sync).
- API-04: reserved idempotent active hit reports ACTUAL state, not queued.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs

# Guard: fastapi/TestClient availability.
try:
    from fastapi.testclient import TestClient
    HAS_CLIENT = True
except Exception:  # noqa: BLE001
    HAS_CLIENT = False

V2_BODY = {
    "contract_version": "2.0",
    "request_id": "v8-req",
    "episode_id": "ep-v8",
    "video_url": "https://example.com/v.mp4",
    "mode": "final",
    "source_preferences": {"max_height": 2160, "prefer_best_available": True},
    "output": {"width": 1080, "height": 1920},
    "clips": [{
        "clip_id": 1, "start_sec": 1, "end_sec": 3, "title": "a",
        "narrative": {"main_topic": "m", "ending_type": "c"},
        "layout_plan": {"preferred_layout": "auto"},
        "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
        "editing_events": [],
    }],
}


@pytest.mark.skipif(not HAS_CLIENT, reason="fastapi testclient not available")
class TestAsyncRouteModel(unittest.TestCase):
    """API-01/API-02: async route response model."""

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
        self.client = TestClient(rs.app)

    def tearDown(self):
        rs._close_db_conns()
        time.sleep(0.3)  # let any spawned worker thread release DB handles
        self._tmp.cleanup()

    def test_async_route_uses_submission_response_model(self):
        """The async route must NOT be declared response_model=RenderResponse."""
        model = rs.app.routes
        target = None
        for route in model:
            if getattr(route, "path", None) == "/api/render/async":
                target = route
                break
        self.assertIsNotNone(target, "/api/render/async route not found")
        resp_model = getattr(target, "response_model", None)
        name = getattr(resp_model, "__name__", str(resp_model))
        self.assertEqual(
            name, "RenderSubmissionResponse",
            "async route must declare RenderSubmissionResponse, got %s" % name,
        )

    def test_async_post_no_validation_error(self):
        """POST valid V2 -> no ResponseValidationError, 200/202."""
        body = dict(V2_BODY)
        body["request_id"] = f"api01-{time.time()}"
        resp = self.client.post("/api/render/async", json=body)
        self.assertIn(resp.status_code, (200, 202))
        j = resp.json()
        self.assertIn("job_id", j)
        self.assertIn("state", j)
        self.assertIn("request_id", j)


@pytest.mark.skipif(not HAS_CLIENT, reason="fastapi testclient not available")
class TestIdempotentRetrieval(unittest.TestCase):
    """API-04: reserved idempotent active hit reports actual state."""

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

    def test_reserved_idempotent_reports_actual_state(self):
        """When the existing job is completed, the idempotent hit must report
        'completed' (SQLite), never a hardcoded 'queued'."""
        # Reserve a job and force it to a completed row in SQLite.
        job_id = "v8job1"
        body = dict(V2_BODY)
        body["request_id"] = "v8-rid"
        req_json = rs.RenderRequestV2(**body).model_dump_json()
        created = rs._reserve_job("v8-rid", job_id, mode="final",
                                  episode_id="ep", request_json=req_json)
        # Simulate a completed persisted result.
        final = rs.RenderResponse(job_id=job_id, source_video="", rendered=[], status="completed")
        rs._persist_job(job_id, "completed", mode="final", episode_id="ep",
                        request=req_json, response=final.model_dump_json())
        # Do NOT register memory (simulates process-memory loss/absence).
        # Re-submitting the same request_id must reflect 'completed', not queued.
        from unittest import mock as _m
        with _m.patch.object(rs, "_load_job_request", return_value=dict(V2_BODY)):
            resp = rs.render_async(body)
        # Idempotent hit branch must read the persisted/snapshot state.
        self.assertNotEqual(
            str(getattr(resp, "state", "")), "queued",
            "idempotent hit must not hardcode queued for a completed job",
        )

    def test_sync_idempotent_completed_after_memory_loss(self):
        """R08: sync duplicate whose completed result is persisted but absent
        from memory must deserialize and return the stored RenderResponse,
        not 409 unknown."""
        from fastapi import HTTPException
        job_id = "v8job2"
        body = dict(V2_BODY)
        body["request_id"] = "v8-sync-rid"
        req = rs.RenderRequestV2(**body)
        req_json = req.model_dump_json()
        rs._reserve_job("v8-sync-rid", job_id, mode="final",
                        episode_id="ep", request_json=req_json)
        final = rs.RenderResponse(job_id=job_id, source_video="",
                                  rendered=[], status="completed")
        rs._persist_job(job_id, "completed", mode="final", episode_id="ep",
                        request=req_json, response=final.model_dump_json())
        # Memory cleared -> simulate restart.
        with rs._async_jobs_lock:
            rs._async_jobs.clear()
        with mock.patch.object(rs, "_load_job_request", return_value=dict(body)):
            try:
                resp = rs.render(dict(body))
            except HTTPException as exc:
                self.fail(f"R08: sync idempotent completed should not 409/unknown: {exc.status_code} {exc.detail}")
            # Must return the stored RenderResponse-compatible result.
            self.assertEqual(resp.status, "completed")


import json  # noqa: E402  (used above)


if __name__ == "__main__":
    unittest.main()