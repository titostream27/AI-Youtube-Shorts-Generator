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

Phase 1 §5.5: validation rules are shared with the TypeScript schema
(src/lib/render/contract.ts in youtube-content-miner) and the neutral JSON
Schema in contracts/render-request-v2.schema.json. Both sides must pass and
reject the same fixtures.
"""
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "2.0"

VALID_MODES = ("preview", "final")


def _is_finite(v: float) -> bool:
    """Brief v5 C-01: reject NaN / +/-Infinity (parity with Zod Number.isFinite)."""
    import math
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
VALID_LAYOUTS = ("auto", "face_crop", "dual_face", "blur_background",
                 "stacked_source", "screen_plus_face")
VALID_EVENT_TYPES = ("emphasis", "punchline", "important_number", "topic_label")

CONTRACT_VERSION = "2.0"
# ── Shared leaf types ──────────────────────────────────────────────────────

class CaptionWord(BaseModel):
    """A single word with precise timing (faster-whisper word timestamps)."""
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    max_height: int = 2160
    prefer_best_available: bool = True


class RenderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = 1080
    height: int = 1920
    fps: Optional[int] = None


class Narrative(BaseModel):
    # Brief v5 C-01: nested models forbid extras (parity with Zod strict).
    model_config = ConfigDict(extra="forbid")
    main_topic: str = ""
    ending_type: str = ""
    hook_end_sec: Optional[float] = None
    payoff_start_sec: Optional[float] = None


class LayoutPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preferred_layout: str = "auto"  # auto|face_crop|dual_face|blur_background|stacked_source|screen_plus_face
    expected_speakers: Optional[int] = None
    allow_split: bool = True
    allow_blur_background: bool = True


class CaptionCue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_sec: float
    end_sec: float
    text: str
    speaker_id: Optional[str] = None
    # Hardening sprint P0.4: canonical word-level timing (when available).
    # If present, the compositor uses these timestamps directly and skips
    # full re-transcription; if absent it falls back to forced alignment.
    words: List[CaptionWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_word_bounds(self) -> "CaptionCue":
        # Hardening sprint P0.4 (parity with Zod cross-field invariant): each
        # word's timing must sit inside its own cue.
        for w in self.words:
            if w.start_sec < self.start_sec or w.end_sec > self.end_sec or w.end_sec <= w.start_sec:
                raise ValueError(
                    f"word [{w.start_sec},{w.end_sec}] invalid for cue [{self.start_sec},{self.end_sec}]"
                )
        return self


class CaptionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str = "en"
    provider: str = "unknown"
    transcript_version: str = ""
    alignment_confidence: float = Field(default=0.0, ge=0, le=1)
    cues: List[CaptionCue] = Field(default_factory=list)
    highlight_terms: List[str] = Field(default_factory=list)


class EditingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_sec: float
    type: str = "emphasis"  # emphasis|punchline|important_number|topic_label
    intensity: float = Field(default=0.5, ge=0, le=1)


class V2Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    # Phase-2 correctness: unknown fields are rejected, not silently dropped —
    # a typo'd key must not be mistaken for a valid contract.
    model_config = ConfigDict(extra="forbid")
    contract_version: str = CONTRACT_VERSION
    request_id: str = ""
    episode_id: str = ""
    video_url: str
    mode: str = "final"  # preview|final
    # Hardening v3 E3: true forces a NEW attempt (history retained).
    force_rerender: bool = False
    source_preferences: SourcePreferences = Field(default_factory=SourcePreferences)
    output: RenderOutput = Field(default_factory=RenderOutput)
    clips: List[V2Clip] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_contract_rules(self):
        """Phase 1 §5.5: enforce the shared contract rules (mirrors the
        TypeScript schema and the JSON Schema in contracts/)."""
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"unsupported contract_version {self.contract_version!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.video_url:
            raise ValueError("video_url must be non-empty")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.output.width <= 0 or self.output.height <= 0:
            raise ValueError("output width and height must be positive")
        seen_ids = set()
        for clip in self.clips:
            # Brief v5 C-01: normalize clip_id to a non-empty string before
            # duplicate detection — numeric 1 and string "1" are the same id.
            norm_id = str(clip.clip_id).strip()
            if not norm_id:
                raise ValueError(f"clip_id must normalize to a non-empty string, got {clip.clip_id!r}")
            # Brief v5 C-01: reject NaN/Infinity in numeric fields.
            for field_name in ("start_sec", "end_sec", "hook_end_sec", "payoff_start_sec"):
                val = getattr(clip, field_name, None)
                if val is not None and not _is_finite(val):
                    raise ValueError(f"clip {norm_id}: {field_name} must be finite")
            if norm_id in seen_ids:
                raise ValueError(f"duplicate clip_id after normalization: {norm_id!r}")
            seen_ids.add(norm_id)
            if clip.start_sec < 0:
                raise ValueError(f"clip {norm_id}: start_sec must be >= 0")
            if clip.end_sec <= clip.start_sec:
                raise ValueError(
                    f"clip {norm_id}: end_sec ({clip.end_sec}) must be > "
                    f"start_sec ({clip.start_sec})"
                )
            if clip.layout_plan.preferred_layout not in VALID_LAYOUTS:
                raise ValueError(
                    f"clip {clip.clip_id}: invalid preferred_layout "
                    f"{clip.layout_plan.preferred_layout!r}"
                )
            for cue in clip.caption_plan.cues:
                if cue.start_sec < clip.start_sec or cue.end_sec > clip.end_sec:
                    raise ValueError(
                        f"clip {clip.clip_id}: caption cue [{cue.start_sec},"
                        f"{cue.end_sec}] outside clip range "
                        f"[{clip.start_sec},{clip.end_sec}]"
                    )
            for ev in clip.editing_events:
                if ev.time_sec < clip.start_sec or ev.time_sec > clip.end_sec:
                    raise ValueError(
                        f"clip {clip.clip_id}: editing event at {ev.time_sec} "
                        f"outside clip range [{clip.start_sec},{clip.end_sec}]"
                    )
                if ev.type not in VALID_EVENT_TYPES:
                    raise ValueError(
                        f"clip {clip.clip_id}: invalid editing event type {ev.type!r}"
                    )
        return self


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


class RenderArtifactResult(BaseModel):
    """Brief v5 6.3 — strict artifact invariant:

    - status=ok        -> video_url required, error absent
    - status=error     -> error required, publishable=false
    - qc_status!=passed in final mode -> publishable=false
    """
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    status: str  # ok | error
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    publishable: bool
    qc_status: str = "unavailable"  # passed | failed | unavailable
    error: Optional[Dict] = None

    @model_validator(mode="after")
    def _validate_artifact_invariants(self) -> "RenderArtifactResult":
        if self.status == "ok":
            if not self.video_url:
                raise ValueError("status=ok requires video_url")
            if self.error is not None:
                raise ValueError("status=ok must not carry error")
        elif self.status == "error":
            if self.error is None:
                raise ValueError("status=error requires error detail")
            if self.publishable:
                raise ValueError("status=error must set publishable=false")
        else:
            raise ValueError(f"status must be 'ok' or 'error', got {self.status!r}")
        if self.qc_status not in ("passed", "failed", "unavailable"):
            raise ValueError(f"qc_status must be passed|failed|unavailable, got {self.qc_status!r}")
        return self


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


class RenderResultClip(BaseModel):
    """One clip result inside the shared RenderResult (hardening v3 E1)."""
    clip_id: Union[str, int]
    status: str  # ok | error
    clip_url: Optional[str] = None
    duration_sec: Optional[float] = None
    error: Optional[str] = None


class RenderResult(BaseModel):
    """Canonical render RESULT shared with the miner (hardening v3 E1).

    Mirrors contracts/render-result-v2.schema.json so the same fixtures pass
    JSON Schema, Zod (miner) and Pydantic (renderer) identically.
    """
    model_config = ConfigDict(extra="forbid")
    contract_version: str = CONTRACT_VERSION
    request_id: str
    episode_id: str
    job_id: Optional[str] = None
    state: str  # completed|partial_failure|failed|cancelled
    error: Optional[str] = None
    source_video: Optional[str] = None
    clips: List[RenderResultClip] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_state(self) -> "RenderResult":
        if self.state not in ("completed", "partial_failure", "failed", "cancelled"):
            raise ValueError(f"invalid render result state: {self.state}")
        return self
