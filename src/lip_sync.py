import argparse
import json
import os
import shutil
import sys
import time
import subprocess
import ffmpeg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import clear_vram, get_vram_info, handle_cuda_oom

try:
    import torch
except ImportError:
    torch = None

def get_video_duration(video_path: str) -> float:
    """Gets exact duration of video file in seconds."""
    try:
        probe = ffmpeg.probe(video_path)
        format_info = probe.get('format', {})
        return float(format_info.get('duration', 0.0))
    except Exception:
        return 0.0

def split_video_and_audio_chunks(video_path: str, audio_path: str, chunk_length_sec: float, temp_chunks_dir: str) -> list[tuple[str, str, int]]:
    """
    Splits video and audio files into smaller equal time chunks (e.g. 10 seconds)
    to prevent GPU VRAM overflow during Wav2Lip processing.
    """
    os.makedirs(temp_chunks_dir, exist_ok=True)
    duration = get_video_duration(video_path)
    
    if duration <= 0.0:
        duration = 10.0

    chunk_info = []
    start_sec = 0.0
    chunk_idx = 0

    print(f"[INFO] Splitting video ({duration:.2f}s) into {chunk_length_sec}s chunks for VRAM safety...")

    while start_sec < duration:
        v_chunk = os.path.join(temp_chunks_dir, f"video_chunk_{chunk_idx:04d}.mp4")
        a_chunk = os.path.join(temp_chunks_dir, f"audio_chunk_{chunk_idx:04d}.wav")

        # Slice video chunk
        (
            ffmpeg
            .input(video_path, ss=start_sec, t=chunk_length_sec)
            .output(v_chunk, vcodec="copy", acodec="copy", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Slice audio chunk
        (
            ffmpeg
            .input(audio_path, ss=start_sec, t=chunk_length_sec)
            .output(a_chunk, ar=16000, ac=1, loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        chunk_info.append((v_chunk, a_chunk, chunk_idx))
        start_sec += chunk_length_sec
        chunk_idx += 1

    print(f"[INFO] Total Chunks Created: {len(chunk_info)}")
    return chunk_info


def process_wav2lip_chunk(v_chunk: str, a_chunk: str, out_chunk: str, checkpoint_path: str, device: str):
    """
    Processes a single 10-second video/audio chunk using Wav2Lip inference.
    """
    # Check if Wav2Lip model checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Wav2Lip model checkpoint not found at: {checkpoint_path}", file=sys.stderr)
        print("[HINT] Download 'wav2lip.pth' or 'wav2lip_gan.pth' into 'models/' folder.", file=sys.stderr)
        print("[FALLBACK] Copying original chunk without lip-sync modification.")
        
        # Fallback mux video + audio
        input_v = ffmpeg.input(v_chunk)
        input_a = ffmpeg.input(a_chunk)
        (
            ffmpeg
            .output(input_v.video, input_a.audio, out_chunk, vcodec="copy", acodec="aac", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return

    # Run Wav2Lip inference on chunk
    try:
        cmd = [
            sys.executable, "-m", "Wav2Lip.inference",
            "--checkpoint_path", checkpoint_path,
            "--face", v_chunk,
            "--audio", a_chunk,
            "--outfile", out_chunk,
            "--nosmooth"
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as e:
        print(f"[WARNING] Wav2Lip inference fallback for chunk: {e}")
        # Mux fallback
        input_v = ffmpeg.input(v_chunk)
        input_a = ffmpeg.input(a_chunk)
        (
            ffmpeg
            .output(input_v.video, input_a.audio, out_chunk, vcodec="copy", acodec="aac", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )


def concatenate_video_chunks(synced_chunks: list[str], output_video: str):
    """
    Concatenates processed video chunks into a single seamless output file using FFmpeg.
    """
    temp_dir = os.path.dirname(synced_chunks[0])
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")

    with open(concat_list_path, "w", encoding="utf-8") as f:
        for chunk in synced_chunks:
            # Normalize path for FFmpeg concat list
            norm_path = os.path.abspath(chunk).replace("\\", "/")
            f.write(f"file '{norm_path}'\n")

    print(f"[INFO] Concatenating {len(synced_chunks)} lip-synced video chunks into final video...")

    try:
        (
            ffmpeg
            .input(concat_list_path, format="concat", safe=0)
            .output(output_video, c="copy", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[SUCCESS] Final Lip-Synced Video saved to: {output_video}")
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"[ERROR] FFmpeg chunk concat failed:\n{stderr_msg}", file=sys.stderr)
        shutil.copyfile(synced_chunks[0], output_video)


def run_lip_sync(
    video_path: str,
    dubbed_audio_path: str,
    output_video: str = None,
    checkpoint_path: str = None,
    chunk_length_sec: float = 10.0,
    skip_lipsync: bool = False,
    device_name: str = "auto"
) -> str:
    """
    Synchronizes speaker lip movements with dubbed audio using Wav2Lip with VRAM-safe chunking.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    if not os.path.exists(dubbed_audio_path):
        raise FileNotFoundError(f"Dubbed audio file not found: {dubbed_audio_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    if output_video is None:
        filename_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_video = os.path.join(output_dir, f"{filename_stem}_lipsynced.mp4")

    # Fast Skip Option
    if skip_lipsync:
        print("\n" + "=" * 60)
        print(" ⏩ SKIP LIP-SYNC FLAG ENABLED: Muxing dubbed audio with video...")
        print("=" * 60)
        
        input_v = ffmpeg.input(video_path)
        input_a = ffmpeg.input(dubbed_audio_path)
        (
            ffmpeg
            .output(input_v.video, input_a.audio, output_video, vcodec="copy", acodec="aac", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[SUCCESS] Fast Dubbed Video saved without lip-sync: {output_video}")
        return output_video

    # Checkpoint resolution
    if checkpoint_path is None:
        model_weights_1 = os.path.join(project_root, "models", "wav2lip_gan.pth")
        model_weights_2 = os.path.join(project_root, "models", "wav2lip.pth")
        
        if os.path.exists(model_weights_1):
            checkpoint_path = model_weights_1
        elif os.path.exists(model_weights_2):
            checkpoint_path = model_weights_2
        else:
            checkpoint_path = model_weights_1

    device = "cuda" if (torch and torch.cuda.is_available() and device_name in ["cuda", "auto"]) else "cpu"
    temp_chunks_dir = os.path.join(project_root, "temp", "lipsync_chunks")

    print("=" * 60)
    print(f"[INFO] Video Input    : {video_path}")
    print(f"[INFO] Dubbed Audio   : {dubbed_audio_path}")
    print(f"[INFO] Wav2Lip Model  : {checkpoint_path}")
    print(f"[INFO] Chunk Size     : {chunk_length_sec} seconds")
    print(f"[INFO] Compute Device : {device.upper()}")
    print(f"[INFO] Initial VRAM   : {get_vram_info()}")
    print("=" * 60)

    # STEP 1: Split video and audio into 10s chunks
    chunks = split_video_and_audio_chunks(video_path, dubbed_audio_path, chunk_length_sec, temp_chunks_dir)

    synced_chunks = []
    total_chunks = len(chunks)
    start_time = time.time()

    # STEP 2: Process each chunk sequentially
    for v_chunk, a_chunk, idx in chunks:
        out_chunk = os.path.join(temp_chunks_dir, f"synced_chunk_{idx:04d}.mp4")
        print(f"[{idx + 1}/{total_chunks}] Processing Wav2Lip chunk {idx:04d}...")
        
        try:
            process_wav2lip_chunk(v_chunk, a_chunk, out_chunk, checkpoint_path, device)
            synced_chunks.append(out_chunk)
        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                handle_cuda_oom(e, f"Wav2Lip Chunk {idx}", "Processing remaining chunks on CPU.")
            else:
                print(f"[ERROR] Chunk {idx} processing failed: {e}")
        finally:
            clear_vram()

    # STEP 3: Concatenate all synced video chunks
    concatenate_video_chunks(synced_chunks, output_video)

    total_duration = time.time() - start_time
    print("-" * 60)
    print(f"[INFO] Wav2Lip Lip-Sync completed in {total_duration / 60:.2f} minutes.")
    print(f"[SUCCESS] Final Lip-Synced Video: {output_video}")
    print(f"[INFO] Final VRAM Status: {get_vram_info()}")
    print("-" * 60)

    return output_video


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize speaker lip movements with dubbed audio using Wav2Lip (VRAM-safe chunking)."
    )
    parser.add_argument(
        "-v", "--video",
        required=True,
        help="Path to original input video file (e.g. input/my_video.mp4)"
    )
    parser.add_argument(
        "-a", "--audio",
        required=True,
        help="Path to dubbed audio WAV file (e.g. temp/full_dubbed_audio.wav)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output video path. Defaults to output/<video>_lipsynced.mp4"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to Wav2Lip checkpoint (models/wav2lip_gan.pth or models/wav2lip.pth)"
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=10.0,
        help="Video chunk duration in seconds for VRAM safety (default: 10.0)"
    )
    parser.add_argument(
        "--skip-lipsync",
        action="store_true",
        help="Skip Lip-Sync processing for fast iteration/testing"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. Default: auto"
    )

    args = parser.parse_args()
    run_lip_sync(
        video_path=args.video,
        dubbed_audio_path=args.audio,
        output_video=args.output,
        checkpoint_path=args.checkpoint,
        chunk_length_sec=args.chunk_sec,
        skip_lipsync=args.skip_lipsync,
        device_name=args.device
    )

if __name__ == "__main__":
    main()
