"""Versioned render contract (Master Task Brief §16-17).

Miner decides WHAT and WHY; renderer decides HOW. This module is the shared
API contract between youtube-content-miner and AI-Youtube-Shorts-Generator.

Contract rules (brief §17):
- Miner sends: clip boundary, narrative structure, caption text/timing,
  highlight terms, preferred layout, whether split is allowed, emphasis
  events, required output quality.
- Renderer decides: face coordinates, camera path, split timing, crop
  position, encoder settings, technical fallback layout.
- Renderer MAY replace the preferred layout if technical quality is too low,
  but MUST report the reason (requested_layout / actual_layout /
  fallback_reason).

v1 (legacy, still accepted): {video_url, clips:[{clip_id,title,start_sec,
  end_sec,captions,hook}], aspect_ratio}
v2: full contract with contract_version="2.0", mode, narrative, layout_plan,
  caption_plan, editing_events, source_preferences, output.
"""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

CONTRACT_VERSION = "2.0"


# ── Shared leaf types ──────────────────────────────────────────────────────

class CaptionWord(BaseModel):
    """A single word with precise timing (faster-whisper word timestamps)."""
    start_sec: float
    end_sec: float
    text: str


class CaptionRequest(BaseModel):
    """A caption line in ABSOLUTE video coordinates (seconds from video start).

    `words` is optional: when present it carries per-word timestamps from
    faster-whisper so the karaoke reveal matches the actual speech rhythm.
    When absent, the render service falls back to evenly distributing the
    line's span across its words.

    `speaker` is optional: when diarization is enabled each line is tagged
    with SPEAKER_00 / SPEAKER_01 / ... and rendered in that speaker's color.
    """
    start_sec: float
    end_sec: float
    text: str
    words: List[CaptionWord] = Field(default_factory=list)
    speaker: str = ""


# ── v2 contract models ─────────────────────────────────────────────────────

class SourcePreferences(BaseModel):
    max_height: int = 2160
    prefer_best_available: bool = True


class RenderOutput(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: Optional[int] = None


class Narrative(BaseModel):
    main_topic: str = ""
    ending_type: str = ""
    hook_end_sec: Optional[float] = None
    payoff_start_sec: Optional[float] = None


class LayoutPlan(BaseModel):
    preferred_layout: str = "auto"  # auto|face_crop|dual_face|blur_background|stacked_source|screen_plus_face
    expected_speakers: Optional[int] = None
    allow_split: bool = True
    allow_blur_background: bool = True


class CaptionCue(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    speaker_id: Optional[str] = None


class CaptionPlan(BaseModel):
    language: str = "en"
    cues: List[CaptionCue] = Field(default_factory=list)
    highlight_terms: List[str] = Field(default_factory=list)


class EditingEvent(BaseModel):
    time_sec: float
    type: str = "emphasis"  # emphasis|punchline|important_number|topic_label
    intensity: float = Field(default=0.5, ge=0, le=1)


class V2Clip(BaseModel):
    clip_id: Union[str, int]
    start_sec: float
    end_sec: float
    title: str = ""
    narrative: Narrative = Field(default_factory=Narrative)
    layout_plan: LayoutPlan = Field(default_factory=LayoutPlan)
    caption_plan: CaptionPlan = Field(default_factory=CaptionPlan)
    editing_events: List[EditingEvent] = Field(default_factory=list)
    # Backward-compat: legacy fields used by v1 callers.
    captions: List[CaptionRequest] = Field(default_factory=list)
    hook: str = ""


class RenderRequestV2(BaseModel):
    contract_version: str = CONTRACT_VERSION
    request_id: str = ""
    episode_id: str = ""
    video_url: str
    mode: str = "final"  # preview|final
    source_preferences: SourcePreferences = Field(default_factory=SourcePreferences)
    output: RenderOutput = Field(default_factory=RenderOutput)
    clips: List[V2Clip] = Field(min_length=1)


# ── Legacy v1 request (still accepted, upgraded internally) ────────────────

class ClipRequest(BaseModel):
    clip_id: int | str
    title: str = ""
    start_sec: float
    end_sec: float
    captions: List[CaptionRequest] = Field(default_factory=list)
    hook: str = ""


class RenderRequest(BaseModel):
    video_url: str
    clips: List[ClipRequest] = Field(min_length=1)
    aspect_ratio: str = "9:16"


# ── Responses ──────────────────────────────────────────────────────────────

class QCDetail(BaseModel):
    score: float = 0
    output_width: int = 1080
    output_height: int = 1920
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    upscale_factor: float = 1.0
    audio_lufs: Optional[float] = None
    audio_true_peak: Optional[float] = None
    audio_sync_ms: Optional[int] = None
    focus_switch_count: int = 0
    focus_ping_pong_detected: bool = False
    random_crop_detected: bool = False
    face_cutoff_ratio: float = 0
    black_frame_ratio: float = 0
    frozen_frame_ratio: float = 0
    caption_overflow: bool = False
    caption_face_collision: bool = False
    warnings: List[str] = Field(default_factory=list)


class RenderArtifact(BaseModel):
    clip_id: Union[str, int]
    status: str  # ok|error|skipped
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error: Optional[str] = None
    duration_sec: Optional[float] = None
    requested_layout: str = "auto"
    actual_layout: str = ""
    fallback_reason: Optional[str] = None
    qc: QCDetail = Field(default_factory=QCDetail)


class SourceInfo(BaseModel):
    width: int = 0
    height: int = 0
    codec: str = ""
    duration_sec: Optional[float] = None


class RenderJobStatus(BaseModel):
    job_id: str
    status: str  # queued|downloading|analysing_source|rendering_preview|rendering_final|quality_check|completed|partial_failure|failed|cancelled
    mode: str = "final"
    source: SourceInfo = Field(default_factory=SourceInfo)
    artifacts: List[RenderArtifact] = Field(default_factory=list)
    error: Optional[str] = None


class RenderResponse(BaseModel):
    job_id: str
    source_video: str
    rendered: List[Dict]
    # Phase 2/4 (brief §23): structured QC artifacts + job metadata exposed on
    # the response so GET /api/render/status/:id can return them without
    # re-parsing the persisted payload.
    artifacts: Optional[List[Dict]] = None
    mode: str = "final"
    source: Optional[Dict] = None
