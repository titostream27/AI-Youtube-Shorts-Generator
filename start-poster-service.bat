@echo off
REM Poster service (Phase 3) - YouTube auto-post. Started via Scheduled Task AelfLab_PosterService.
cd /d D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator
set POSTER_PORT=8085
set RENDER_OUTPUT_DIR=D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator\rendered
".venv\Scripts\python.exe" poster_service.py >> D:\homelab\hermes-workspace\AI-Youtube-Shorts-Generator\poster-service.log 2>&1
