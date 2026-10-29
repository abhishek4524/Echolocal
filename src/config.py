"""
src/config.py
Centralized configuration for EchoLocal pipeline.

All hardcoded magic numbers and tuneable parameters live here.
Import `cfg` in any module instead of scattering literals everywhere.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# ── Project root (two levels up from src/) ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing parameters."""

    sample_rate: int = 22050                # Hz — matches XTTS-v2 native rate
    target_lufs: float = -16.0             # Loudness normalization target (EBU R128)
    crossfade_ms: int = 75                 # Crossfade between TTS segments (ms)
    bg_volume: float = 0.35               # Background music mix level
    bg_duck_threshold: float = 0.25       # Sidechain duck threshold (0–1)
    bg_duck_ratio: float = 1.8            # Sidechain compression ratio
    tts_min_speed: float = 0.85           # Minimum TTS playback speed ratio
    tts_max_speed: float = 1.30           # Maximum TTS playback speed ratio
    silence_pad_ms: int = 150             # Padding before each TTS segment


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and parameters."""

    whisper_model: str = "small"           # faster-whisper model size
    whisper_compute: str = "int8"          # Quantization: int8 | float16 | float32
    whisper_device: str = "auto"           # 'auto' | 'cuda' | 'cpu'

    translation_model_en_hi: str = "Helsinki-NLP/opus-mt-en-hi"
    translation_model_hi_en: str = "Helsinki-NLP/opus-mt-hi-en"

    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    diarize_n_speakers: Optional[int] = None   # None = auto-detect
    diarize_min_speakers: int = 1
    diarize_max_speakers: int = 4


@dataclass(frozen=True)
class PipelineConfig:
    """End-to-end pipeline defaults."""

    max_video_resolution: str = "1080"     # Max height for YouTube downloads
    batch_extensions: tuple = (".mp4", ".mkv", ".avi", ".mov", ".webm")
    output_suffix: str = "_dubbed"         # Appended to output filenames
    segment_gap_s: float = 0.05           # Min silence gap between speaker segments


@dataclass(frozen=True)
class PathConfig:
    """Default directory layout (relative to project root)."""

    input_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "input")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    temp_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "temp")


@dataclass(frozen=True)
class EchoLocalConfig:
    """Top-level configuration object — import this."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Module-level singleton — `from src.config import cfg`
cfg = EchoLocalConfig()
