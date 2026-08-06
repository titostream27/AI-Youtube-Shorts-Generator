"""Shared contract fixture tests (Phase 1 §5.5 / §5.7 items 9-10).

Loads the neutral fixtures from contracts/fixtures and asserts the Python
validator accepts every valid fixture and rejects every invalid one.

Run with:
    .venv/Scripts/python.exe -m pytest test_contract_fixtures.py -q
"""
import json
import os
import sys
import unittest
from pathlib import Path

from render_contract import RenderRequestV2

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "content-miner" / "contracts"
VALID_DIR = CONTRACTS_DIR / "fixtures" / "valid"
INVALID_DIR = CONTRACTS_DIR / "fixtures" / "invalid"


class TestSharedFixturesPython(unittest.TestCase):
    def test_valid_fixtures_parse(self):
        # Phase-2 correctness: missing shared fixtures must FAIL CI, not skip.
        self.assertTrue(
            VALID_DIR.exists(),
            "contracts/fixtures/valid not present — run the miner build first",
        )
        v2_files = [p for p in VALID_DIR.glob("*.json") if "v2" in p.name and not p.name.startswith("render-result")]
        self.assertGreater(len(v2_files), 0, "expected at least one valid v2 fixture")
        for path in v2_files:
            with self.subTest(fixture=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                req = RenderRequestV2(**payload)
                self.assertEqual(req.contract_version, "2.0")

    def test_invalid_fixtures_rejected(self):
        if not INVALID_DIR.exists():
            self.fail("contracts/fixtures/invalid not present — run the miner build first")
        invalid_files = sorted(INVALID_DIR.glob("*.json"))
        self.assertGreater(len(invalid_files), 0, "expected at least one invalid fixture")
        for path in invalid_files:
            with self.subTest(fixture=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                with self.assertRaises(Exception, msg=f"{path.name} should be invalid"):
                    RenderRequestV2(**payload)


class TestRenderResultFixtureParity(unittest.TestCase):
    """Hardening v3 E1 (#26/#29): the neutral render-result fixtures must pass
    the Pydantic RenderResult validator identically to JSON Schema / Zod."""

    def test_valid_render_result_fixtures_parse(self):
        from render_contract import RenderResult
        files = [p for p in VALID_DIR.glob("render-result*.json")]
        self.assertGreater(len(files), 0, "expected render-result valid fixtures")
        for path in files:
            with self.subTest(fixture=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                res = RenderResult(**payload)
                self.assertEqual(res.contract_version, "2.0")

    def test_invalid_render_result_fixtures_rejected(self):
        from render_contract import RenderResult
        files = [p for p in INVALID_DIR.glob("render-result*.json")]
        self.assertGreater(len(files), 0, "expected render-result invalid fixtures")
        for path in files:
            with self.subTest(fixture=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                with self.assertRaises(Exception, msg=f"{path.name} should be invalid"):
                    RenderResult(**payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
