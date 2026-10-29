"""
tests/conftest.py
Shared pytest fixtures and configuration.

Fixtures defined here are automatically available to all test modules.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Add src/ to sys.path once for all tests ───────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def add_src_to_path():
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ── Shared heavy-import stubs ─────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def stub_ffmpeg(monkeysession):
    """
    Prevent subprocess.run calls to ffmpeg from actually executing during tests.
    Tests that need real ffmpeg should opt-out with @pytest.mark.no_ffmpeg_stub.
    """
    pass  # Individual tests can patch subprocess.run as needed


@pytest.fixture
def sample_segments():
    """Return a minimal list of transcription segments for testing."""
    return [
        {"start": 0.0,  "end": 2.5,  "text": "Hello, how are you?",      "speaker": "SPEAKER_00"},
        {"start": 3.0,  "end": 5.0,  "text": "I am fine, thank you.",    "speaker": "SPEAKER_01"},
        {"start": 5.5,  "end": 8.0,  "text": "That is great to hear.",   "speaker": "SPEAKER_00"},
    ]


@pytest.fixture
def sample_hindi_segments():
    """Hindi segment list for translation tests."""
    return [
        {"start": 0.0, "end": 2.0, "text": "नमस्ते, आप कैसे हैं?",   "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 4.5, "text": "मैं ठीक हूँ, धन्यवाद।",  "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def tmp_video(tmp_path):
    """Create a minimal dummy .mp4 file for I/O tests (not a real video)."""
    p = tmp_path / "test_video.mp4"
    p.write_bytes(b"\x00" * 1024)
    return p


@pytest.fixture
def tmp_audio(tmp_path):
    """Create a minimal dummy .wav file for audio tests."""
    import struct
    # 44-byte WAV header (1 second, mono, 22050 Hz, 16-bit)
    p = tmp_path / "test_audio.wav"
    num_samples = 22050
    data = b"\x00\x00" * num_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, 22050, 44100, 2, 16,
        b"data", len(data)
    )
    p.write_bytes(header + data)
    return p
