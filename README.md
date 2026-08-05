# AI YouTube Shorts Generator

**The media execution worker for a personal Podcast Opportunity Miner.**
This repository renders approved clip boundaries into vertical, caption-ready
9:16 shorts — it does **not** decide which moments are viral, change clip
boundaries, or re-rank candidates. Discovery, transcript, boundary
refinement, clip scoring and review all live in
[`youtube-content-miner`](https://github.com/titostream27/youtube-content-miner).

The renderer owns: source download/cache, face + active-speaker tracking,
crop and virtual-camera planning, layout execution, captions, audio,
thumbnails, encoding, and technical quality control.

![longshorts](https://github.com/user-attachments/assets/3f5d1abf-bf3b-475f-8abf-5e253003453a)

## Why This Architecture?

| Concern | Owner |
|---|---|
| What to clip and why (discovery, scoring, ranking, review) | `youtube-content-miner` |
| How approved boundaries are presented (crop, layout, captions, encode, QC) | **this repo** |
| Virality ranking / boundary changes | **nowhere in this repo** (removed in Phase 2) |

## Features

- **🎬 Approved Boundaries In, Vertical Out**: consumes `RenderRequestV2`
  (v1 legacy accepted) and returns per-clip artifacts + structured QC
- **🤖 Face-Aware Reframing**: YuNet DNN dual-mode tracking, active-speaker
  follow, reaction-split support with caption-safe placement
- **🎤 Whisper Karaoke + Speaker Diarization**: word-level caption reveal
  synced to real speech; per-speaker colors
- **🧩 Technical QC**: audio loudness/sync, black/frozen frame detection,
  focus-switch / ping-pong / face-cutoff tracking stats
- **💾 Restart-Safe Jobs**: SQLite job store (WAL), persisted idempotency by
  `request_id`, queued cancellation, retry history via `parent_job_id`
- **🧪 Deterministic Tests**: shared contract fixtures + synthetic visual
  fixtures (no YouTube needed)

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt
pip install -r requirements-local.txt   # yt-dlp, faster-whisper, opencv

# 2. Run the render worker
.venv/Scripts/python.exe render_service.py
# FastAPI on http://127.0.0.1:8084
```

Point `youtube-content-miner` at it (`RENDER_BASE_URL`) and the pipeline
handles the rest.

---

## Installation (Self-Hosted)

### Prerequisites

- Python 3.10+
- `ffmpeg` on PATH
- LLM/whisper keys only when the miner's caption/transcript path requires them
  (the renderer itself uses `faster-whisper` locally for karaoke timing)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/titostream27/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-local.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   RENDER_HOST=127.0.0.1
   RENDER_PORT=8084
   RENDER_OUTPUT_DIR=rendered
   RENDER_WHISPER_MODEL=base
   RENDER_WHISPER_DEVICE=auto
   ```

## Usage

### Run the render service (production path)

This repository is a media execution worker: it renders approved boundaries
that `youtube-content-miner` sends. It is not a CLI highlight finder.

```bash
.venv/Scripts/python.exe render_service.py
# FastAPI on http://127.0.0.1:8084
```

Endpoints:

| Route | Purpose |
|---|---|
| `POST /api/render/async` | Queue a render job (v1 or v2 contract), returns `job_id` immediately |
| `GET /api/render/status/{job_id}` | Poll a job's canonical state |
| `POST /api/render/jobs/{job_id}/cancel` | Cancel a queued job (active render -> 409) |
| `POST /api/render/jobs/{job_id}/retry` | Retry a failed job (new `job_id`, records `parent_job_id`) |
| `POST /api/render` | Synchronous render (same identity rule) |
| `GET /api/render/health` | Operational readiness (DB, queue, ffmpeg, disk, contract) |

A minimal v2 request:

```bash
curl -X POST http://127.0.0.1:8084/api/render/async \
  -H "Content-Type: application/json" \
  -d '{"contract_version":"2.0","request_id":"req-1","episode_id":"ep-1",
       "video_url":"https://www.youtube.com/watch?v=VIDEO_ID","mode":"final",
       "clips":[{"clip_id":1,"start_sec":10,"end_sec":45,"title":"t",
                "narrative":{"main_topic":"m","ending_type":"c",
                             "hook_end_sec":null,"payoff_start_sec":null},
                "caption_plan":{"cues":[]}}]}'
```

The miner (`youtube-content-miner`) builds these payloads through
`buildRenderContract` / its render routes — you normally never hand-write one.

### Local renderer development

Run the render service against a local video for fast iteration:

```bash
RENDER_OUTPUT_DIR=/tmp/rendered .venv/Scripts/python.exe render_service.py
```

### Fixtures

Synthetic deterministic fixture clips (no YouTube needed) for visual
regression:

```bash
.venv/Scripts/python.exe scripts/make_visual_fixtures.py fixtures/visual
```

## How It Works (production path)

1. **Receive**: `render_async` parses v1/v2 manually (a v2 payload is never
   mis-parsed as v1), checks idempotency by `request_id`, creates `job_id`
   once, persists `queued`.
2. **Download**: source is downloaded once per job and cached by video id.
3. **Reframe**: each approved clip is cut and reframed to 9:16 with face
   tracking (YuNet DNN dual-mode), camera path, and layout honoring the
   miner's `preferred_layout` (fallback with reported reason).
4. **Captions**: whisper word-level karaoke + speaker diarization; caption
   placement avoids faces and shifts center during reaction-split windows.
5. **QC**: technical quality gate (audio, sync, black/frozen frames) + focus
   switch / ping-pong / face-cutoff tracking stats.
6. **Persist**: every transition is written to SQLite under the SAME
   `job_id`; the response returns artifacts per clip.

The renderer does **not** discover highlights, change clip boundaries, or
re-rank candidates — all of that lives in `youtube-content-miner`.

## Output

The render service returns a `RenderResponse` with one artifact per clip:

```json
{
  "job_id": "abc123",
  "source_video": "...",
  "rendered": [{"clip_id": 1, "status": "ok", "clip_url": "...", "duration_sec": 35.0}],
  "artifacts": [{"clip_id": 1, "status": "ok", "video_url": "...", "qc": {"score": 88, "focus_switch_count": 2}}]
}
```

Poll `GET /api/render/status/{job_id}` until `state` is `completed`, then read
the persisted `response` from the SQLite `render_jobs` table (survives
restarts).

## Configuration

Set via environment variables (see `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `RENDER_HOST` / `RENDER_PORT` | `127.0.0.1` / `8084` | Bind address |
| `RENDER_OUTPUT_DIR` | `rendered` | Output root |
| `RENDER_JOB_DB` | `<output>/render_jobs.db` | SQLite job store |
| `RENDER_FORMAT` | `2160` | Source download resolution |
| `RENDER_VIDEO_CRF` / `RENDER_VIDEO_PRESET` | `17` / `slow` | Final H.264 encode |
| `RENDER_WHISPER_MODEL` / `RENDER_WHISPER_DEVICE` | `base` / `cpu` | Caption transcription |
| `RENDER_DIARIZE` | `1` | Speaker diarization |
| `RENDER_BUILD_ID` | `0.1.0` | Health endpoint build id |

## Two-repository architecture (Phase 1)

This repository is the **media execution worker**: it renders **approved
boundaries** and decides **how to present them** (camera, layout, captions,
audio, thumbnails, encoding, QC). The intelligence/control plane is
[`youtube-content-miner`](https://github.com/titostream27/youtube-content-miner),
which decides **what to clip and why** (discovery, transcript, candidates,
boundaries, scoring, review, RenderRequest creation).

The renderer must **not** independently discover highlights, change clip
boundaries, or re-rank candidates in the integrated production path. The
standalone highlight pipeline (`main.py` / MuAPI clipper) was **removed in
Phase 2** — all highlight discovery, boundary selection and virality ranking
lives in `youtube-content-miner`.

### Shared contract

The canonical contract is owned by `youtube-content-miner` in its
[`contracts/`](https://github.com/titostream27/youtube-content-miner/tree/main/contracts)
directory:

- `render-request-v2.schema.json`, `render-result-v2.schema.json` — neutral
  JSON Schema source of truth.
- `fixtures/valid/`, `fixtures/invalid/` — shared fixtures both sides must
  pass/reject identically.

This repo implements the Python side (`render_contract.py`: `RenderRequestV2`,
pydantic + `model_validator`). Contract tests load the shared fixtures from
`../content-miner/contracts/fixtures` (`test_contract_fixtures.py`).
`contract_version` must be exactly `"2.0"`; unsupported versions fail
explicitly.

### Job states

One canonical vocabulary everywhere (memory, SQLite, API, logs):

```
queued ─► downloading ─► analysing ─► rendering ─► quality_check ─► completed
```
Terminal: `failed | partial_failure | cancelled | orphaned`. A job waiting
for the render lock is `queued`. `_render(request, job_id)` receives its
`job_id` from the job service and never generates a replacement — the same id
appears in the API response, SQLite row, output directory, and artifact URLs.

### Persistence, retry, cancellation

- SQLite `render_jobs`: WAL mode + busy timeout; additive idempotent
  migration (`request_id`, `parent_job_id`, `attempt`, `started_at`,
  `finished_at`, `last_error_stage` + indexes). Existing records preserved.
- Idempotency: a resubmitted `request_id` returns the existing job — survives
  renderer restart.
- Retry (`POST /api/render/jobs/{job_id}/retry`): one new `job_id`, records
  `parent_job_id`, increments `attempt`.
- Cancellation (`POST /api/render/jobs/{job_id}/cancel`): **queued** jobs are
  cancelled and never enter rendering. Active (mid-FFmpeg) cancellation is
  NOT supported in Phase 1 — cancelling a `rendering` job returns HTTP 409.

### Health

`GET /api/render/health` reports operational readiness without downloading a
video or loading models: service status + build id (`RENDER_BUILD_ID`),
SQLite read/write probe, queue depth + active job id, ffmpeg/ffprobe
availability, output dir writability + free disk, contract version, last
sanitized persistence error, and `rendering_available_when_persistence_degraded`.

### Contract & lifecycle tests

```bash
.venv/Scripts/python.exe -m pytest test_render_contract.py test_contract_fixtures.py test_job_lifecycle.py -q
```

## Project Structure

```
AI-Youtube-Shorts-Generator/
├── render_service.py            FastAPI render worker (production entry)
├── render_contract.py           shared v2 contract (pydantic side)
├── poster_service.py            YouTube upload worker (OAuth2)
├── quality_gate.py              technical QC (audio/sync/frames)
├── audio_master.py              audio loudness/normalization
├── visual_effects.py            crop quality, layout, color, emphasis
├── requirements.txt             core deps
├── .env.example
├── scripts/
│   └── make_visual_fixtures.py  deterministic fixture clip generator
├── fixtures/visual/             generated synthetic clips (git-ignored *.mp4)
└── shorts_generator/
    ├── config.py                env / settings (render + local backends)
    └── local/                   cutting + reframing backends (offline)
        ├── downloader.py        yt-dlp download
        ├── transcriber.py       faster-whisper transcription
        ├── llm.py               OpenAI or Gemini client selector
        └── clipper.py           ffmpeg cut + OpenCV/YuNet vertical crop
```

## Troubleshooting

### Whisper produced no segments
The render service logs `whisper -> 0 segments` for a clip. Set
`RENDER_WHISPER_MODEL` to a larger model (e.g. `small`/`medium`) or check the
source audio track. Transcription is per-clip inside the service; there is no
CLI flag.

### Render service health says degraded
`GET /api/render/health` reports `status: degraded` when the SQLite store or
output dir is not writable. Check `db.error` / `output.error` in the payload;
persistence failures are also surfaced via `last_persist_error`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Related Projects

- [AI Influencer Generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)
- [Text to Video AI](https://github.com/SamurAIGPT/Text-To-Video-AI)
- [Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)
- [AI B-roll Generator](https://github.com/Anil-matcha/AI-B-roll)
- [No-code YouTube Shorts Generator](https://www.vadoo.tv/clip-youtube-video)
- [ai-creator-academy](https://github.com/Anil-matcha/ai-creator-academy) — free curriculum teaching creators how to monetize AI-generated shorts and video content
