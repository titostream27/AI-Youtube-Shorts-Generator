"""Brief v8 C05 — test(renderer): persistence-failure and metadata-preservation
lifecycle cases (RED on v7, GREEN after C06).

Findings:
- R03/STATE-01: on a terminal PersistenceError the worker must NOT fabricate
  memory state 'failed' when SQLite may still hold an active state. It must
  mirror the durable (SQLite) state and set persistence_degraded=True.
- STATE-02: when the terminal transition is lost, memory mirrors the durable
  winner.
- R09/STATE-03: success/error paths preserve request_id/mode/episode/attempt/
  parent metadata.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_service as rs


def _req_body(rid):
    return {
        "contract_version": "2.0",
        "request_id": rid,
        "episode_id": "ep",
        "video_url": "https://example.com/v.mp4",
        "mode": "final",
        "clips": [{
            "clip_id": 1, "start_sec": 1, "end_sec": 3, "title": "a",
            "narrative": {"main_topic": "m", "ending_type": "c"},
            "layout_plan": {"preferred_layout": "auto"},
            "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
            "editing_events": [],
        }],
    }


class StateV8Base(unittest.TestCase):
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


class TestPersistExceptionNoFabricatedState(StateV8Base):
    """R03/STATE-01: persist exception must NOT invent a durable failed."""

    def test_worker_does_not_fabricate_failed_on_persist_error(self):
        job_id = "persist-x"
        body = _req_body("persist-x")
        req_json = rs.RenderRequestV2(**body).model_dump_json()
        rs._reserve_job("persist-x", job_id, mode="final", episode_id="ep",
                        request_json=req_json)
        rs._register_job_memory(job_id, "persist-x", "final", "ep")
        fake_outcome = rs.RenderOutcome(
            rs.RenderResponse(job_id=job_id, source_video="", rendered=[], status="completed"),
            "completed",
        )
        with mock.patch.object(rs, "_load_job_request", return_value=body), \
             mock.patch.object(rs, "_render", return_value=fake_outcome), \
             mock.patch.object(rs, "_persist_terminal_via_transition",
                               side_effect=rs.PersistenceError("db locked")):
            rs._process_queued_job(job_id)
        # What SQLite actually holds at this point is the worker's last
        # committed active stage (downloading/analysing/rendering) — memory
        # must mirror THAT, never a fabricated 'failed'.
        durable = rs._load_job(job_id)
        with rs._async_jobs_lock:
            mem = rs._async_jobs.get(job_id, {})
        # Memory must NOT claim a fabricated terminal failed.
        self.assertNotEqual(
            mem.get("state"), "failed",
            "memory must not fabricate failed when SQLite holds active",
        )
        # It must mirror the durable (SQLite) state.
        self.assertEqual(mem.get("state"), durable["status"],
                         "memory must mirror durable state")
        # And flag persistence degradation.
        self.assertTrue(mem.get("persistence_degraded"),
                        "must set persistence_degraded=True")


class TestMetadataPreserved(StateV8Base):
    """R09-STATE-03: metadata survives every terminal path."""

    def test_success_path_keeps_metadata(self):
        job_id = "meta-ok"
        body = _req_body("meta-ok")
        req_json = rs.RenderRequestV2(**body).model_dump_json()
        rs._reserve_job("meta-ok", job_id, mode="final", episode_id="ep",
                        request_json=req_json)
        rs._register_job_memory(job_id, "meta-ok", "final", "ep")
        with rs._async_jobs_lock:
            rs._async_jobs[job_id]["attempt"] = 3
            rs._async_jobs[job_id]["parent_job_id"] = "parent-9"
        fake_outcome = rs.RenderOutcome(
            rs.RenderResponse(job_id=job_id, source_video="", rendered=[], status="completed"),
            "completed",
        )
        with mock.patch.object(rs, "_load_job_request", return_value=body), \
             mock.patch.object(rs, "_render", return_value=fake_outcome), \
             mock.patch.object(rs, "_persist_terminal_via_transition", return_value=True):
            rs._process_queued_job(job_id)
        with rs._async_jobs_lock:
            mem = rs._async_jobs.get(job_id, {})
        self.assertEqual(mem.get("request_id"), "meta-ok")
        self.assertEqual(mem.get("mode"), "final")
        self.assertEqual(mem.get("episode_id"), "ep")
        self.assertEqual(mem.get("attempt"), 3)
        self.assertEqual(mem.get("parent_job_id"), "parent-9")


if __name__ == "__main__":
    unittest.main()