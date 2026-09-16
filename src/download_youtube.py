import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import ensure_ffmpeg_in_path

ensure_ffmpeg_in_path()

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sanitize_filename(title: str, max_length: int = 120) -> str:
    """Convert a video title into a safe filesystem filename.

    Removes / replaces characters that are illegal on Windows/Linux/macOS,
    collapses consecutive spaces, and trims to max_length.
    """
    # Replace common problematic characters with safe equivalents
    replacements = {
        "/": "-", "\\": "-", ":": "-", "*": "",
        "?": "", '"': "'", "<": "", ">": "", "|": "-",
        "\n": " ", "\r": " ", "\t": " ",
    }
    safe = title
    for bad, good in replacements.items():
        safe = safe.replace(bad, good)

    # Strip any remaining non-printable / control characters
    safe = re.sub(r"[\x00-\x1f\x7f]", "", safe)
    # Collapse runs of whitespace
    safe = re.sub(r"\s+", " ", safe).strip()
    # Trim to max_length (leave room for extension)
    safe = safe[:max_length]
    return safe or "video"


def is_youtube_url(url: str) -> bool:
    """Return True if the string looks like a YouTube URL."""
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=",
        r"(?:https?://)?youtu\.be/",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/",
        r"(?:https?://)?(?:www\.)?youtube\.com/live/",
        r"(?:https?://)?(?:m\.)?youtube\.com/watch\?v=",
    ]
    return any(re.search(p, url) for p in patterns)


# --------------------------------------------------------------------------- #
# Core download function
# --------------------------------------------------------------------------- #

def download_youtube_video(
    url: str,
    output_dir: str = None,
    max_height: int = 1080,
    prefer_mp4: bool = True,
) -> str:
    """Download a YouTube video and return the local file path.

    Args:
        url:        Full YouTube video URL.
        output_dir: Folder to save the video. Defaults to <project>/input/.
        max_height: Maximum vertical resolution (default 1080p).
        prefer_mp4: If True, prefer mp4 container; falls back to mkv/webm.

    Returns:
        Absolute path to the downloaded video file.

    Raises:
        ImportError:  yt-dlp is not installed.
        ValueError:   URL is not a recognised YouTube URL.
        RuntimeError: Download failed (private / unavailable / age-restricted).
    """
    if yt_dlp is None:
        raise ImportError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )

    if not is_youtube_url(url):
        raise ValueError(
            f"URL does not look like a YouTube link: {url}\n"
            "Please provide a valid YouTube video URL."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if output_dir is None:
        output_dir = os.path.join(project_root, "input")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 – fetch video metadata first (no download) to get clean title
    # ------------------------------------------------------------------ #
    print(f"[INFO] Fetching video metadata: {url}")
    meta_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        err = str(e).lower()
        if "private" in err:
            raise RuntimeError(
                "This video is private. Please use a public YouTube video."
            )
        elif "age" in err or "sign in" in err or "login" in err:
            raise RuntimeError(
                "This video is age-restricted and cannot be downloaded without login."
            )
        elif "unavailable" in err or "not available" in err:
            raise RuntimeError(
                "This video is unavailable (removed, geo-blocked, or does not exist)."
            )
        else:
            raise RuntimeError(f"Failed to fetch video info: {e}")

    raw_title = info.get("title", "video")
    safe_title = sanitize_filename(raw_title)
    duration_sec = info.get("duration", 0)

    print(f"[INFO] Video Title     : {raw_title}")
    print(f"[INFO] Safe Filename   : {safe_title}")
    print(f"[INFO] Duration        : {duration_sec // 60}m {duration_sec % 60}s")

    # ------------------------------------------------------------------ #
    # Step 2 – build format selector
    #   • Try best mp4 up to max_height
    #   • Fallback to best video+audio merged up to max_height
    # ------------------------------------------------------------------ #
    if prefer_mp4:
        format_sel = (
            f"bestvideo[height<={max_height}][ext=mp4]+"
            f"bestaudio[ext=m4a]/"
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/"
            "best"
        )
    else:
        format_sel = (
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/"
            "best"
        )

    # Output template — yt-dlp fills in extension automatically
    output_template = os.path.join(output_dir, f"{safe_title}.%(ext)s")

    ydl_opts = {
        "format": format_sel,
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "noprogress": False,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    # ------------------------------------------------------------------ #
    # Step 3 – download
    # ------------------------------------------------------------------ #
    print(f"\n[INFO] Downloading video (max {max_height}p) -> {output_dir}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        err = str(e).lower()
        if "private" in err:
            raise RuntimeError("Video is private and cannot be downloaded.")
        elif "age" in err or "sign in" in err or "login" in err:
            raise RuntimeError(
                "Video is age-restricted. Cannot download without authentication."
            )
        elif "unavailable" in err or "not available" in err:
            raise RuntimeError(
                "Video is unavailable (removed, geo-blocked, or does not exist)."
            )
        else:
            raise RuntimeError(f"Download failed: {e}")

    # ------------------------------------------------------------------ #
    # Step 4 – find the downloaded file and return its path
    # ------------------------------------------------------------------ #
    for ext in ("mp4", "mkv", "webm", "avi", "mov"):
        candidate = os.path.join(output_dir, f"{safe_title}.{ext}")
        if os.path.exists(candidate):
            file_size_mb = os.path.getsize(candidate) / (1024 * 1024)
            print(f"[SUCCESS] Downloaded: {candidate} ({file_size_mb:.1f} MB)")
            return candidate

    # Fallback: scan the output_dir for a recently created video file
    video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    candidates = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if os.path.splitext(f)[1].lower() in video_exts
            and safe_title[:30] in f
    ]
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        print(f"[SUCCESS] Downloaded: {candidates[0]}")
        return candidates[0]

    raise RuntimeError(
        f"Download appeared to succeed but output file not found in: {output_dir}"
    )


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Download a YouTube video at up to 1080p for EchoLocal pipeline processing."
    )
    parser.add_argument(
        "url",
        help="YouTube video URL (e.g. https://www.youtube.com/watch?v=xxxxx)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output folder for the downloaded video. Defaults to input/"
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum video height in pixels (default: 1080)"
    )

    args = parser.parse_args()
    try:
        path = download_youtube_video(
            url=args.url,
            output_dir=args.output_dir,
            max_height=args.max_height,
        )
        print(f"\n[DONE] Video ready at: {path}")
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
