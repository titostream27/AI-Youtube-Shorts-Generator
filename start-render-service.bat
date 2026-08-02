@echo off
REM Render service (AI-Youtube-Shorts-Generator fork) - hybrid shorts renderer.
REM Runs the FastAPI service on port 8084 (host loopback only).
REM Autostart via Windows Scheduled Task: AelfLab_RenderService

cd /d D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator
set RENDER_PORT=8084
set RENDER_OUTPUT_DIR=D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator\rendered
set RENDER_PATH_PREFIX=short
set RENDER_FACE_ZOOM=0.6
set RENDER_FORMAT=1080
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set RENDER_DEBUG_TRACK=1
set RENDER_SPLIT=1
set RENDER_SPLIT_FADE_S=0.3
set RENDER_SPLIT_HOLD_S=2.5
set RENDER_SPLIT_MOUTH_DELTA=0.15
set RENDER_SPLIT_SINGLE_S=0.4

".venv\Scripts\python.exe" render_service.py >> D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator\render-service.log 2>&1
