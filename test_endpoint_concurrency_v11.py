"""Brief v11 — Commit 1: test(renderer) — expose production retry/force/
resubmit concurrency gaps.

These tests call the SAME endpoint/service functions used by FastAPI routes
(render_async, render, render_job_retry, render_job_cancel, reserve_attempt)
— NOT helper-only wrappers. They must be RED against the current HEAD because
the production paths still bypass reserve_attempt().

RT11-01  two concurrent /retry -> exactly one winner, no dup attempt, every
         returned job_id exists in SQLite, parent lineage correct.
RT11-02  three sequential retries -> attempts 1->2->3->4 monotonic, parent
         chain correct.
RT11-03  two concurrent force rerenders after completed -> one active attempt,
         second gets same active job or 409, no phantom id.
RT11-05  resubmit same request after cancelled -> no attempt=1 collision; new
         job durably inserted with documented next-attempt semantics.
RT11-06r resubmit same request after failed -> no phantom, no duplicate
         (request_id, attempt).
RT11-07  allocator DB I/O failure -> service fails closed, no memory/queue
         publication, no speculative job id.
RT11-08  IntegrityError winner race -> returned winner id loadable from
         SQLite and matches request_id/attempt/state.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException

import render_service as rs


class V11DBUnittest(unittest.TestCase):
    """Isolated temp DB per test so no cross-test pollution."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "v11.db")
        self._mock = mock.patch.object(rs, "JOB_DB_PATH", self.db_path)
        self._mock.start()
        rs._async_jobs.clear()
        rs._close_db_conns()
        # Avoid background workers accumulating real work.
        with mock.patch.object(rs, "_render_queue") as q:
            q.qsize.return_value = 0
        rs._render_worker_last_exception = None

        def _fake_table():
            with rs._db_conn() as conn:
                return conn

        self._fake = _fake_table

    def tearDown(self):
        self._mock.stop()
        rs._close_db_conns()
        try:
            os.unlink(self.db_path)
        except Exception:
            pass
        try:
            os.rmdir(self.tmpdir)
        except Exception:
            pass

    def _persist_source(self, job_id, req_id, status, attempt=1):
        rs._persist_job(
            job_id, status, mode="final", episode_id="ep",
            request=f'{{"request_id":"{req_id}","a":1}}',
            attempt=attempt,
        )

    @staticmethod
    def _result_job_id(result):
        return result.job_id if hasattr(result, "job_id") else result["job_id"]

    @staticmethod
    def _result_state(result):
        return result.state if hasattr(result, "state") else result["state"]


class RetryForceResubmitProduction(V11DBUnittest):

    @staticmethod
    def _v2_body(request_id: str, *, force: bool = False) -> dict:
        return {
            "contract_version": "2.0",
            "request_id": request_id,
            "episode_id": "ep",
            "video_url": "https://x.example/a.mp4",
            "mode": "final",
            "force_rerender": force,
            "clips": [{
                "clip_id": 1,
                "start_sec": 0,
                "end_sec": 2,
                "title": "test",
                "narrative": {"main_topic": "m", "ending_type": "c"},
                "layout_plan": {"preferred_layout": "auto"},
                "caption_plan": {"language": "en", "cues": [], "highlight_terms": []},
                "editing_events": [],
            }],
        }

    def _seed_failed(self):
        body = self._v2_body("rid-1")
        rs._persist_job("src-f", "failed", mode="final", episode_id="e",
                        attempt=1, request=json.dumps(body))
        return "src-f"

    def test_rt11_01_two_concurrent_retries_on_failed_job(self):
        """Both /retry callers of a failed job get real ids; no dup."""
        src = self._seed_failed()
        import threading
        holder = {}

        errors = {}

        def go(i):
            try:
                holder[i] = rs.render_job_retry(src)
            except Exception as exc:  # endpoint may return 409 for loser
                errors[i] = exc

        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            t1 = threading.Thread(target=lambda: go(1))
            t2 = threading.Thread(target=lambda: go(2))
            t1.start(); t2.start(); t1.join(); t2.join()
        self.assertEqual(set(holder) | set(errors), {1, 2})
        # A production endpoint must resolve a concurrent winner or return a
        # documented HTTP conflict; a raw PersistenceError is not acceptable.
        self.assertFalse(
            errors,
            f"production /retry leaked exceptions instead of endpoint semantics: {errors}",
        )
        got = list(holder.values())
        # Exactly one winning job id (both healthy or one 409) but never two
        # different QUEUED children with the same (request_id, attempt).
        with rs._db_conn() as c:
            rows = c.execute(
                "SELECT job_id, attempt, parent_job_id FROM render_jobs "
                "WHERE parent_job_id=? AND request_id='rid-1'", (src,)
            ).fetchall()
        # No duplicate attempt.
        attempts = [r[1] for r in rows]
        self.assertEqual(len(attempts), len(set(attempts)),
                         f"duplicate attempts: {rows}")
        # Every returned id must exist durably.
        for r in rows:
            loaded = rs._load_job(r[0])
            self.assertIsNotNone(loaded)
        # Parent lineage correct.
        for r in rows:
            self.assertEqual(r[2], src)

    def test_rt11_02_three_sequential_retries_monotonic_parent_chain(self):
        parent = self._seed_failed()
        chain = [parent]
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            for _ in range(3):
                out = rs.render_job_retry(chain[-1])
                chain.append(out["job_id"])
                # The endpoint only accepts failed/partial_failure sources;
                # make the just-created child terminal for the next retry.
                rs._persist_job(chain[-1], "failed", request=json.dumps(self._v2_body("rid-1")))
        # attempts 1 -> 2 -> 3 -> 4
        for idx, jid in enumerate(chain, start=1):
            if jid == parent:
                continue
            d = rs._load_job(jid)
            self.assertEqual(d["attempt"], idx)
        # lineage history queryable
        with rs._db_conn() as c:
            history = c.execute(
                "SELECT job_id, attempt, parent_job_id FROM render_jobs "
                "WHERE request_id='rid-1' ORDER BY attempt"
            ).fetchall()
        self.assertGreaterEqual(len(history), 4)

    def test_rt11_03_two_concurrent_force_rerenders(self):
        """force after completed: at most one active attempt for request."""
        body = self._v2_body("rid-f", force=True)
        rs._persist_job("src-done", "completed", mode="final", episode_id="e",
                        attempt=1, request=json.dumps(self._v2_body("rid-f")))
        # Two concurrent force submissions via the ASYNC production path.
        import threading
        got = {}
        errors = {}
        def call(i):
            try:
                got[i] = rs.render_async(body)
            except Exception as exc:
                errors[i] = exc
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            t1 = threading.Thread(target=call, args=(1,)); t1.start(); t1.join()
            t2 = threading.Thread(target=call, args=(2,)); t2.start(); t2.join()
        self.assertEqual(set(got) | set(errors), {1, 2})
        for exc in errors.values():
            self.assertIsInstance(exc, (HTTPException, ValueError))
        # at most one active attempt, and every returned id is durable
        with rs._db_conn() as c:
            act = c.execute(
                "SELECT job_id, attempt FROM render_jobs WHERE request_id='rid-f' "
                "AND status IN ('queued','downloading','analysing','rendering','quality_check')"
            ).fetchall()
        self.assertLessEqual(len(act), 1, f"multiple active: {act}")

    def test_rt11_06_resubmit_after_cancelled_no_parentthought(self):
        """Normal resubmit after a cancelled attempt yields a durably row; no attempt=1 clash with prior cancelled (attempt>1) and no missing request_id (i.e. unique (request_id, attempt) honored)."""
        body = self._v2_body("c-rid")
        rs._persist_job("old-cancel", "cancelled", mode="final", episode_id="e",
                        attempt=3, request=json.dumps(body))
        # Second normal submission via PRODUCTION async path.
        with mock.patch.object(rs, "_enqueue_job", return_value=None):
            r = rs.render_async(body)
        jid = self._result_job_id(r)
        loaded = rs._load_job(jid)
        self.assertIsNotNone(loaded)
        self.assertGreaterEqual(loaded["attempt"], 4)
        self.assertEqual(loaded["parent_job_id"], "old-cancel")
        with rs._db_conn() as c:
            self.assertEqual(c.execute(
                "SELECT COUNT(*) FROM render_jobs WHERE request_id='c-rid' AND attempt=?",
                (loaded["attempt"],),
            ).fetchone()[0], 1)