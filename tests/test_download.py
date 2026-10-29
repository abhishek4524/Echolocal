"""
tests/test_download.py
Unit tests for src/download_youtube.py — filename sanitization & URL validation.

Run with:  pytest tests/ -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _stub_yt_dlp():
    yt_dlp = types.ModuleType("yt_dlp")
    yt_dlp.YoutubeDL = MagicMock()
    sys.modules.setdefault("yt_dlp", yt_dlp)
    return yt_dlp


@pytest.fixture(scope="module")
def download_module():
    _stub_yt_dlp()
    import download_youtube
    return download_youtube


# ═════════════════════════════════════════════════════════════════════════════
# sanitize_filename
# ═════════════════════════════════════════════════════════════════════════════

class TestSanitizeFilename:

    def test_removes_windows_forbidden_chars(self, download_module):
        """Characters forbidden in Windows filenames must be stripped/replaced."""
        raw = 'Video: "Title" | Part <1> / Test?'
        safe = download_module.sanitize_filename(raw)
        for ch in r'\/:*?"<>|':
            assert ch not in safe, f"Forbidden char '{ch}' still in: {safe}"

    def test_preserves_alphanumeric(self, download_module):
        raw = "HelloWorld123"
        safe = download_module.sanitize_filename(raw)
        assert "HelloWorld123" in safe

    def test_max_length_respected(self, download_module):
        """Long titles must be truncated to avoid OS filename limits."""
        raw = "A" * 300
        safe = download_module.sanitize_filename(raw)
        assert len(safe) <= 200, f"Filename too long: {len(safe)} chars"

    def test_empty_string_returns_fallback(self, download_module):
        """Empty or all-special input should not return empty string."""
        raw = "???"
        safe = download_module.sanitize_filename(raw)
        assert isinstance(safe, str)
        assert len(safe) > 0

    def test_strips_leading_trailing_spaces_dots(self, download_module):
        raw = "  .My Video Title.  "
        safe = download_module.sanitize_filename(raw)
        assert not safe.startswith(" ")
        assert not safe.endswith(" ")

    def test_unicode_title(self, download_module):
        """Hindi / Unicode titles should survive sanitization."""
        raw = "यह एक वीडियो है — टेस्ट"
        safe = download_module.sanitize_filename(raw)
        assert isinstance(safe, str)
        assert len(safe) > 0

    def test_returns_string_type(self, download_module):
        assert isinstance(download_module.sanitize_filename("test"), str)


# ═════════════════════════════════════════════════════════════════════════════
# is_youtube_url  (if exported)
# ═════════════════════════════════════════════════════════════════════════════

class TestIsYouTubeUrl:

    def test_standard_youtube_url(self, download_module):
        if not hasattr(download_module, "is_youtube_url"):
            pytest.skip("is_youtube_url not exported")
        assert download_module.is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtu_be_short(self, download_module):
        if not hasattr(download_module, "is_youtube_url"):
            pytest.skip("is_youtube_url not exported")
        assert download_module.is_youtube_url("https://youtu.be/dQw4w9WgXcQ")

    def test_non_youtube_returns_false(self, download_module):
        if not hasattr(download_module, "is_youtube_url"):
            pytest.skip("is_youtube_url not exported")
        assert not download_module.is_youtube_url("https://vimeo.com/12345")

    def test_local_path_returns_false(self, download_module):
        if not hasattr(download_module, "is_youtube_url"):
            pytest.skip("is_youtube_url not exported")
        assert not download_module.is_youtube_url("/home/user/video.mp4")


# ═════════════════════════════════════════════════════════════════════════════
# download_youtube_video — error handling (mocked)
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadErrorHandling:

    def test_private_video_raises_or_returns_none(self, download_module):
        """Private video should raise a clear exception, not crash silently."""
        import yt_dlp as mock_ydl
        mock_ydl.YoutubeDL.return_value.__enter__ = MagicMock(
            side_effect=Exception("Video unavailable")
        )
        mock_ydl.YoutubeDL.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(Exception):
            download_module.download_youtube_video(
                "https://www.youtube.com/watch?v=PRIVATE",
                output_dir=Path("/tmp/test_output")
            )
