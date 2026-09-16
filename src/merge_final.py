import argparse
import json
import os
import shutil
import sys
import wave
import numpy as np
import ffmpeg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import ensure_ffmpeg_in_path

ensure_ffmpeg_in_path()

def get_media_duration(file_path: str) -> float:
    """Gets total duration of media file in seconds using ffprobe."""
    try:
        probe = ffmpeg.probe(file_path)
        format_info = probe.get('format', {})
        return float(format_info.get('duration', 0.0))
    except Exception:
        return 0.0

def adjust_audio_speed(input_wav: str, output_wav: str, speed_factor: float):
    """
    Adjusts audio playback speed using FFmpeg atempo filter.
    speed_factor > 1.0 speeds up audio (shortens duration).
    speed_factor < 1.0 slows down audio (lengthens duration).
    """
    try:
        (
            ffmpeg
            .input(input_wav)
            .filter('atempo', speed_factor)
            .output(output_wav, loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error:
        shutil.copyfile(input_wav, output_wav)

def read_wav_pcm(wav_path: str):
    """Reads WAV file returning sample_rate, channels, and numpy int16 audio array."""
    with wave.open(wav_path, 'rb') as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_bytes = wf.readframes(nframes)
        
        dtype = np.int16 if sampwidth == 2 else np.int8
        audio_data = np.frombuffer(raw_bytes, dtype=dtype)
        
        if nchannels > 1:
            # Convert stereo to mono by averaging channels
            audio_data = audio_data.reshape(-1, nchannels).mean(axis=1).astype(dtype)
            
        return framerate, audio_data

def mix_voice_and_background_ffmpeg(
    speech_wav: str,
    background_wav: str,
    output_mixed_wav: str,
    bg_volume: float = 0.35
) -> str:
    """
    Mixes cloned voice WAV track with Demucs background WAV track using FFmpeg.
    Resamples both streams to 44.1kHz stereo, applies volume scaling and gentle,
    transparent sidechain compression so background music remains clearly audible at all times.
    """
    if not background_wav or not os.path.exists(background_wav) or bg_volume <= 0.0:
        print("[INFO] Background track not provided or volume is 0. Using voice-only audio.")
        shutil.copyfile(speech_wav, output_mixed_wav)
        return output_mixed_wav

    bg_vol = max(0.0, min(1.0, bg_volume))
    print(f"[INFO] Mixing Demucs background track (music + SFX) (Volume={bg_vol:.2f})...")

    # Gentle sidechain ducking filter (threshold=0.25, ratio=1.8, release=500)
    # Background music is gently ducked by ~3-4 dB when speech occurs, but NEVER muted completely.
    try:
        in_v = ffmpeg.input(speech_wav)
        in_b = ffmpeg.input(background_wav)

        bg_vol_filter = in_b.filter('volume', bg_vol)
        bg_ducked = ffmpeg.filter(
            [bg_vol_filter, in_v],
            'sidechaincompress',
            threshold=0.25,
            ratio=1.8,
            attack=80,
            release=500
        )
        mixed = ffmpeg.filter(
            [in_v, bg_ducked],
            'amix',
            inputs=2,
            duration='first',
            dropout_transition=2
        )

        (
            ffmpeg
            .output(mixed, output_mixed_wav, ac=2, ar=44100, loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[SUCCESS] FFmpeg background music mixing & transparent ducking completed -> {output_mixed_wav}")
        return output_mixed_wav

    except Exception as e:
        print(f"[WARNING] Sidechain ducking filter failed ({e}). Attempting standard amix mix...")

    # Fallback FFmpeg mix without sidechain ducking
    try:
        in_v = ffmpeg.input(speech_wav)
        in_b = ffmpeg.input(background_wav)

        bg_vol_filter = in_b.filter('volume', bg_vol)
        mixed = ffmpeg.filter([in_v, bg_vol_filter], 'amix', inputs=2, duration='first')

        (
            ffmpeg
            .output(mixed, output_mixed_wav, ac=2, ar=44100, loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[SUCCESS] Standard FFmpeg background music mix completed -> {output_mixed_wav}")
        return output_mixed_wav

    except Exception as e2:
        print(f"[ERROR] FFmpeg audio mixing failed: {e2}. Using voice track only.", file=sys.stderr)
        shutil.copyfile(speech_wav, output_mixed_wav)
        return output_mixed_wav



def merge_tts_and_mux_video(
    video_path: str,
    manifest_json: str,
    background_audio: str = None,
    output_video: str = None,
    min_speed: float = 0.85,
    max_speed: float = 1.30,
    bg_volume: float = 0.35
) -> str:
    """
    Places TTS audio segments on target timestamps with atempo speed-sync,
    mixes with Demucs background track (music + SFX) via FFmpeg audio ducking,
    and muxes with original video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(manifest_json):
        raise FileNotFoundError(f"Manifest JSON not found: {manifest_json}")

    with open(manifest_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[ERROR] No TTS segments found in manifest file.", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    temp_dir = os.path.join(project_root, "temp", "synced_segments")
    os.makedirs(temp_dir, exist_ok=True)

    video_duration = get_media_duration(video_path)
    if video_duration <= 0.0:
        video_duration = max([seg.get("end", 0.0) for seg in segments]) + 2.0

    print("=" * 60)
    print(f"[INFO] Video File      : {video_path}")
    print(f"[INFO] Total Duration  : {video_duration:.2f} seconds")
    print(f"[INFO] Manifest File   : {manifest_json}")
    if background_audio:
        print(f"[INFO] Background Track: {background_audio} (Vol: {bg_volume:.2f})")
    else:
        print(f"[INFO] Background Track: None (Voice-only mode)")
    print(f"[INFO] Speed Clamping  : [{min_speed}x -> {max_speed}x]")
    print("=" * 60)

    target_sr = 24000  # Default XTTS sample rate
    total_samples = int(video_duration * target_sr) + (target_sr * 5)
    full_speech_audio = np.zeros(total_samples, dtype=np.int16)

    print(f"[INFO] Syncing {len(segments)} TTS voice segments...")

    for idx, seg in enumerate(segments):
        audio_file = seg.get("audio_file")
        if not audio_file or not os.path.exists(audio_file):
            print(f"[WARNING] Segment {idx} audio file missing. Skipping.")
            continue

        start_time = seg.get("start", 0.0)
        end_time = seg.get("end", start_time + 1.0)
        target_dur = max(0.5, end_time - start_time)

        with wave.open(audio_file, 'rb') as wf:
            gen_dur = wf.getnframes() / float(wf.getframerate())

        raw_speed_ratio = gen_dur / target_dur if target_dur > 0 else 1.0
        clamped_speed = max(min_speed, min(max_speed, raw_speed_ratio))

        synced_wav_path = os.path.join(temp_dir, f"synced_{idx:04d}.wav")

        if abs(clamped_speed - 1.0) > 0.03:
            adjust_audio_speed(audio_file, synced_wav_path, clamped_speed)
        else:
            shutil.copyfile(audio_file, synced_wav_path)

        sr, pcm_data = read_wav_pcm(synced_wav_path)
        target_sr = sr

        # ------------------------------------------------------------------ #
        # Apply 75ms crossfade (fade-in & fade-out) on segment boundaries
        # Prevents abrupt cuts, pops, and clicks between segments
        # ------------------------------------------------------------------ #
        fade_ms = 75
        fade_samples = min(int(sr * (fade_ms / 1000.0)), len(pcm_data) // 2)

        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            pcm_float = pcm_data.astype(np.float32)
            pcm_float[:fade_samples] *= fade_in
            pcm_float[-fade_samples:] *= fade_out
            pcm_data = np.clip(pcm_float, -32768, 32767).astype(np.int16)

        # ------------------------------------------------------------------ #
        # Prevent segment spillover into the next speaker's time window.
        # Eliminates garbled audio disturbance when multiple speakers talk.
        # ------------------------------------------------------------------ #
        if idx < len(segments) - 1:
            next_start_s = segments[idx + 1].get("start", end_time + 1.0)
            max_allowed_len = max(int(0.3 * sr), int((next_start_s - start_time) * sr))
            if len(pcm_data) > max_allowed_len:
                pcm_data = pcm_data[:max_allowed_len]
                if fade_samples > 0 and len(pcm_data) > fade_samples:
                    pcm_float = pcm_data.astype(np.float32)
                    pcm_float[-fade_samples:] *= fade_out
                    pcm_data = np.clip(pcm_float, -32768, 32767).astype(np.int16)

        start_sample = int(start_time * sr)
        end_sample = start_sample + len(pcm_data)

        if end_sample > len(full_speech_audio):
            full_speech_audio = np.pad(full_speech_audio, (0, end_sample - len(full_speech_audio)))

        # Clean max-envelope placement to eliminate clipping distortion between overlapping segment edges
        target_slice = full_speech_audio[start_sample:end_sample]
        overlap_mask = target_slice != 0
        blended = pcm_data[0:len(target_slice)].copy()
        if np.any(overlap_mask):
            prev_vals = target_slice[overlap_mask]
            curr_vals = blended[overlap_mask]
            blended[overlap_mask] = np.where(np.abs(curr_vals) >= np.abs(prev_vals), curr_vals, prev_vals)

        full_speech_audio[start_sample:end_sample] = blended

        start_fmt = f"{int(start_time // 60):02d}:{start_time % 60:05.2f}"
        print(f"  Segment [{start_fmt}] Target: {target_dur:.2f}s | Gen: {gen_dur:.2f}s | Speed: {clamped_speed:.2f}x | Fade: {fade_ms}ms")

    # Save speech-only track
    speech_only_path = os.path.join(project_root, "temp", "speech_only.wav")
    with wave.open(speech_only_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(full_speech_audio.tobytes())

    print(f"[INFO] Speech-only track saved to: {speech_only_path}")

    # Mix speech-only track with Demucs background track using FFmpeg
    combined_audio_path = os.path.join(project_root, "temp", "full_dubbed_audio.wav")
    mix_voice_and_background_ffmpeg(
        speech_wav=speech_only_path,
        background_wav=background_audio,
        output_mixed_wav=combined_audio_path,
        bg_volume=bg_volume
    )

    print(f"[INFO] Final combined audio track ready at: {combined_audio_path}")

    # Output video resolution
    if output_video is None:
        filename_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_video = os.path.join(output_dir, f"{filename_stem}_dubbed.mp4")
    else:
        out_dir = os.path.dirname(os.path.abspath(output_video))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Muxing final audio track into video: {output_video}")

    try:
        input_v = ffmpeg.input(video_path)
        input_a = ffmpeg.input(combined_audio_path)

        (
            ffmpeg
            .output(
                input_v.video,
                input_a.audio,
                output_video,
                vcodec='copy',
                acodec='aac',
                shortest=None,
                loglevel="error"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[SUCCESS] Final dubbed video created successfully: {output_video}")
        return output_video
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"[ERROR] FFmpeg video muxing failed:\n{stderr_msg}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Merge TTS segment audio files with background music and mux with original video."
    )
    parser.add_argument(
        "-v", "--video",
        required=True,
        help="Path to original video file (e.g. input/sample.mp4)"
    )
    parser.add_argument(
        "-m", "--manifest",
        required=True,
        help="Path to TTS manifest JSON file (e.g. temp/tts_segments/manifest.json)"
    )
    parser.add_argument(
        "-b", "--background",
        default=None,
        help="Optional path to Demucs background audio track (temp/separated/.../background.wav)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output video file path. Defaults to output/<video_name>_dubbed.mp4"
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=0.85,
        help="Minimum allowed atempo speed factor (default: 0.85)"
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.30,
        help="Maximum allowed atempo speed factor (default: 1.30)"
    )
    parser.add_argument(
        "--bg-volume",
        type=float,
        default=0.35,
        help="Volume multiplier for background music track (default: 0.35)"
    )

    args = parser.parse_args()
    merge_tts_and_mux_video(
        video_path=args.video,
        manifest_json=args.manifest,
        background_audio=args.background,
        output_video=args.output,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        bg_volume=args.bg_volume
    )

if __name__ == "__main__":
    main()
