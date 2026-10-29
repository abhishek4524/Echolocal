"""
tests/test_diarize.py
Unit tests for src/diarize.py — acoustic speaker clustering logic.

Run with:  pytest tests/ -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _stub_audio_deps():
    """Stub librosa, numpy, sklearn, torch for headless CI."""
    mocks = {
        "librosa": MagicMock(),
        "numpy": MagicMock(),
        "sklearn": MagicMock(),
        "sklearn.cluster": MagicMock(),
        "sklearn.preprocessing": MagicMock(),
        "torch": MagicMock(),
        "pyannote": MagicMock(),
        "pyannote.audio": MagicMock(),
    }
    for name, mock in mocks.items():
        sys.modules.setdefault(name, mock)

    # numpy needs real array-like behavior for some checks
    import numpy as np_mock
    np_mock.ndarray = object
    np_mock.float32 = float
    np_mock.int16 = int


_stub_audio_deps()


@pytest.fixture(scope="module")
def diarize_module():
    import diarize
    return diarize


# ═════════════════════════════════════════════════════════════════════════════
# Segment label assignment
# ═════════════════════════════════════════════════════════════════════════════

class TestSegmentLabeling:

    def test_speaker_labels_are_strings(self, diarize_module):
        """Whatever the diarizer returns, speaker labels must be strings."""
        if not hasattr(diarize_module, "assign_speaker_labels"):
            pytest.skip("assign_speaker_labels not exported")
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 2.5, "end": 4.0, "text": "World"},
        ]
        labels = [0, 1]
        result = diarize_module.assign_speaker_labels(segments, labels)
        for seg in result:
            assert isinstance(seg.get("speaker", ""), str)

    def test_single_speaker_all_same_label(self, diarize_module):
        if not hasattr(diarize_module, "assign_speaker_labels"):
            pytest.skip("assign_speaker_labels not exported")
        segments = [
            {"start": 0.0, "end": 2.0, "text": "A"},
            {"start": 2.0, "end": 4.0, "text": "B"},
        ]
        labels = [0, 0]
        result = diarize_module.assign_speaker_labels(segments, labels)
        speakers = {s["speaker"] for s in result}
        assert len(speakers) == 1


# ═════════════════════════════════════════════════════════════════════════════
# n_speakers auto-detect
# ═════════════════════════════════════════════════════════════════════════════

class TestNSpeakersAutoDetect:

    def test_n_speakers_bounds(self, diarize_module):
        """Auto-detected speaker count should be between 1 and 8."""
        if not hasattr(diarize_module, "estimate_n_speakers"):
            pytest.skip("estimate_n_speakers not exported")
        # Mock MFCC features (5 frames × 13 features)
        mock_features = [[float(i + j) for j in range(13)] for i in range(5)]
        n = diarize_module.estimate_n_speakers(mock_features, min_speakers=1, max_speakers=4)
        assert 1 <= n <= 4
