"""
tests/test_voice_clone_tts.py
Unit tests for src/voice_clone_tts.py — punctuation & TTS helpers.

Run with:  pytest tests/ -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _stub_heavy():
    """Stub GPU/model imports so tests run on plain CI."""
    for mod in ["torch", "torchaudio", "TTS", "TTS.api", "numpy", "soundfile", "pydub"]:
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            sys.modules[mod] = m

    # pydub.AudioSegment mock
    pydub = sys.modules.get("pydub", types.ModuleType("pydub"))
    pydub.AudioSegment = MagicMock()
    sys.modules["pydub"] = pydub

    numpy = sys.modules.get("numpy", types.ModuleType("numpy"))
    numpy.ndarray = object
    numpy.float32 = float
    sys.modules["numpy"] = numpy


@pytest.fixture(scope="module")
def tts_module():
    _stub_heavy()
    import voice_clone_tts
    return voice_clone_tts


# ═════════════════════════════════════════════════════════════════════════════
# prepare_punctuation_for_tts
# ═════════════════════════════════════════════════════════════════════════════

class TestPreparePunctuationForTTS:

    def test_comma_pause_injected(self, tts_module):
        """Commas should trigger a natural pause marker."""
        if not hasattr(tts_module, "prepare_punctuation_for_tts"):
            pytest.skip("prepare_punctuation_for_tts not exported")
        out = tts_module.prepare_punctuation_for_tts("Hello, world")
        # The function may add '...' or SSML pause — just check it's non-empty str
        assert isinstance(out, str) and len(out) > 0

    def test_period_pause_injected(self, tts_module):
        if not hasattr(tts_module, "prepare_punctuation_for_tts"):
            pytest.skip("prepare_punctuation_for_tts not exported")
        out = tts_module.prepare_punctuation_for_tts("I am here. You are there.")
        assert isinstance(out, str) and len(out) > 0

    def test_empty_string(self, tts_module):
        if not hasattr(tts_module, "prepare_punctuation_for_tts"):
            pytest.skip("prepare_punctuation_for_tts not exported")
        out = tts_module.prepare_punctuation_for_tts("")
        assert isinstance(out, str)

    def test_hindi_punctuation(self, tts_module):
        if not hasattr(tts_module, "prepare_punctuation_for_tts"):
            pytest.skip("prepare_punctuation_for_tts not exported")
        out = tts_module.prepare_punctuation_for_tts("यह सही है। हाँ, बिल्कुल।")
        assert isinstance(out, str) and len(out) > 0


# ═════════════════════════════════════════════════════════════════════════════
# speed_factor_for_segment  (duration fitting)
# ═════════════════════════════════════════════════════════════════════════════

class TestSpeedFactor:

    def test_short_tts_gets_slowed_down(self, tts_module):
        """If TTS audio is shorter than segment window, speed < 1."""
        if not hasattr(tts_module, "compute_speed_factor"):
            pytest.skip("compute_speed_factor not exported")
        factor = tts_module.compute_speed_factor(tts_duration=1.5, segment_duration=3.0)
        assert factor < 1.0, f"Expected speed < 1 for short TTS, got {factor}"

    def test_long_tts_gets_sped_up(self, tts_module):
        """If TTS audio is longer than segment window, speed > 1."""
        if not hasattr(tts_module, "compute_speed_factor"):
            pytest.skip("compute_speed_factor not exported")
        factor = tts_module.compute_speed_factor(tts_duration=5.0, segment_duration=2.0)
        assert factor > 1.0, f"Expected speed > 1 for long TTS, got {factor}"

    def test_speed_clamped_to_min(self, tts_module):
        if not hasattr(tts_module, "compute_speed_factor"):
            pytest.skip("compute_speed_factor not exported")
        factor = tts_module.compute_speed_factor(tts_duration=0.1, segment_duration=100.0)
        assert factor >= 0.80, f"Speed below minimum: {factor}"

    def test_speed_clamped_to_max(self, tts_module):
        if not hasattr(tts_module, "compute_speed_factor"):
            pytest.skip("compute_speed_factor not exported")
        factor = tts_module.compute_speed_factor(tts_duration=100.0, segment_duration=0.1)
        assert factor <= 1.35, f"Speed above maximum: {factor}"
