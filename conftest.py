"""Brief Renderer V12R — shared pytest configuration.

V12R-F03: in CI the visual dependencies are MANDATORY. A missing OpenCV or
ffmpeg/ffprobe binary must fail the whole run (setup failure), never skip
visual suites. Locally (no CI env) the enforcement is skipped so pure-logic
development stays unhindered; the visual tests themselves hard-import cv2.
"""

import os
import shutil
import sys

import pytest


def pytest_configure(config):
    if not os.environ.get("CI"):
        return
    try:
        import cv2  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise pytest.UsageError(
            f"V12R-F03: opencv unavailable in CI ({exc}) — install "
            "requirements-ci.txt; visual suites must NOT be skipped."
        ) from exc
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise pytest.UsageError(
                f"V12R-F03: '{binary}' not found in CI PATH — a missing "
                "media binary must fail setup, not skip tests."
            )