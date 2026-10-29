"""
tests/test_config.py
Unit tests for src/config.py — validates config dataclass defaults.

Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module")
def config_module():
    import config
    return config


class TestEchoLocalConfig:

    def test_cfg_singleton_exists(self, config_module):
        assert config_module.cfg is not None

    def test_audio_config_defaults(self, config_module):
        a = config_module.cfg.audio
        assert a.sample_rate == 22050
        assert -25.0 <= a.target_lufs <= -10.0, "LUFS target out of reasonable range"
        assert 0 < a.crossfade_ms <= 500
        assert 0.0 < a.bg_volume <= 1.0

    def test_model_config_defaults(self, config_module):
        m = config_module.cfg.model
        assert m.whisper_model in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        assert m.whisper_compute in {"int8", "float16", "float32", "int8_float16"}

    def test_pipeline_config_defaults(self, config_module):
        p = config_module.cfg.pipeline
        assert p.max_video_resolution.isdigit()
        assert int(p.max_video_resolution) >= 480

    def test_path_config_paths_are_path_objects(self, config_module):
        paths = config_module.cfg.paths
        assert isinstance(paths.input_dir, Path)
        assert isinstance(paths.output_dir, Path)
        assert isinstance(paths.models_dir, Path)

    def test_tts_speed_range_valid(self, config_module):
        a = config_module.cfg.audio
        assert a.tts_min_speed < a.tts_max_speed
        assert 0.5 <= a.tts_min_speed <= 1.0
        assert 1.0 <= a.tts_max_speed <= 2.0

    def test_config_is_frozen(self, config_module):
        """Frozen dataclass — mutation should raise FrozenInstanceError."""
        with pytest.raises(Exception):
            config_module.cfg.audio.sample_rate = 999  # type: ignore[misc]
