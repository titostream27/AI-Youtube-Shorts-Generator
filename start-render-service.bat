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

".venv\Scripts\python.exe" render_service.py
