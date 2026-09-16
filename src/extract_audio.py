import argparse
import os
import shutil
import sys
import ffmpeg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import ensure_ffmpeg_in_path

def check_ffmpeg():
    """Verify if ffmpeg executable is available in PATH or common paths."""
    if ensure_ffmpeg_in_path():
        return True
    
    print("[ERROR] FFmpeg is not installed or not found in system PATH.", file=sys.stderr)
    print("[HINT] Please download FFmpeg for Windows and add its 'bin' directory to system environment PATH.", file=sys.stderr)
    return False

def extract_audio(input_file: str, output_file: str = None) -> str:
    """
    Extracts audio from video file and converts it to 16kHz mono WAV format suitable for Whisper.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input video file not found: {input_file}")
    
    if not check_ffmpeg():
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    if output_file is None:
        filename_stem = os.path.splitext(os.path.basename(input_file))[0]
        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        output_file = os.path.join(temp_dir, f"{filename_stem}.wav")
    else:
        out_dir = os.path.dirname(os.path.abspath(output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Processing Video: {input_file}")
    print(f"[INFO] Output WAV Path: {output_file}")
    print("[INFO] Extracting audio (16kHz, mono, PCM WAV)...")

    try:
        (
            ffmpeg
            .input(input_file)
            .output(
                output_file,
                ar=16000,
                ac=1,
                acodec='pcm_s16le',
                loglevel="error"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[SUCCESS] Audio extracted successfully: {output_file}")
        return output_file
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"[ERROR] FFmpeg processing failed:\n{stderr_output}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Extract 16kHz mono WAV audio from video files (mp4/mkv/mov) for Whisper STT."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input video file (e.g., input/sample.mp4)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional custom output path for WAV file. Defaults to temp/<filename>.wav"
    )

    args = parser.parse_args()
    extract_audio(input_file=args.input, output_file=args.output)

if __name__ == "__main__":
    main()
