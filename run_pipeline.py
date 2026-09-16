import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.gpu_utils import clear_vram, get_vram_info, handle_cuda_oom

from src.download_youtube import download_youtube_video, is_youtube_url
from src.extract_audio import extract_audio
from src.separate_audio import separate_vocals_and_background
from src.transcribe import transcribe_audio
from src.diarize import run_diarization
from src.translate import translate_transcript
from src.generate_subtitles import generate_subtitles_from_json
from src.voice_clone_tts import generate_voice_cloned_tts
from src.merge_final import merge_tts_and_mux_video
from src.lip_sync import run_lip_sync

def run_single_video_pipeline(
    video_path: str,
    output_video: str = None,
    whisper_model: str = "small",
    demucs_model: str = "htdemucs",
    src_lang: str = "auto",
    tgt_lang: str = "auto",
    translation_engine: str = "auto",
    ref_audio: str = None,
    hf_token: str = None,
    device: str = "auto",
    low_vram: bool = True,
    skip_lipsync: bool = False,
    chunk_sec: float = 10.0,
    bg_volume: float = 0.85
) -> dict:
    """Runs the complete EchoLocal Video Dubbing, Subtitle & Lip-Sync Pipeline for a single video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    start_time = time.time()

    print("\n" + "=" * 75)
    print(f" 🎬 PROCESSING VIDEO: {os.path.basename(video_path)}")
    print("=" * 75)

    clear_vram()

    # STEP 1: Audio Extraction
    print(">>> STEP 1/9: Extracting Audio from Video...")
    extracted_wav = extract_audio(input_file=video_path)
    if low_vram: clear_vram()

    # STEP 2: Demucs Vocal & Background Separation
    print(">>> STEP 2/9: Separating Vocals & Background (Demucs)...")
    vocals_wav, background_wav = separate_vocals_and_background(
        input_wav=extracted_wav,
        model_name=demucs_model,
        device_name=device
    )
    if low_vram: clear_vram()

    # STEP 3: Whisper Transcription
    print(">>> STEP 3/9: Transcribing Vocals (faster-whisper)...")
    transcript_json = transcribe_audio(
        input_file=vocals_wav,
        model_name=whisper_model,
        device_name=device,
        language=src_lang if src_lang != "auto" else None
    )
    if low_vram: clear_vram()

    # STEP 4: Speaker Diarization
    print(">>> STEP 4/9: Speaker Diarization (pyannote.audio)...")
    diarized_json = run_diarization(
        audio_wav=vocals_wav,
        transcript_json=transcript_json,
        hf_token=hf_token,
        device_name=device
    )
    if low_vram: clear_vram()

    # STEP 5: Translation
    print(">>> STEP 5/9: Translating Transcript (IndicTrans2/NLLB)...")
    translated_json = translate_transcript(
        input_file=diarized_json,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        engine=translation_engine,
        device=device
    )
    if low_vram: clear_vram()

    # STEP 6: SRT Subtitle Generation
    print(">>> STEP 6/9: Generating .SRT Subtitle Files...")
    subtitle_files = generate_subtitles_from_json(input_json=translated_json)

    # STEP 7: Multi-Speaker Voice Clone TTS
    print(">>> STEP 7/9: Multi-Speaker Voice-Cloned TTS Generation (Coqui XTTS-v2)...")
    tts_reference_audio = ref_audio if ref_audio else vocals_wav
    tts_manifest = generate_voice_cloned_tts(
        input_json=translated_json,
        ref_audio=tts_reference_audio,
        device_name=device
    )
    if low_vram: clear_vram()

    # STEP 8: Merge Audio Tracks
    print(">>> STEP 8/9: Syncing Voice Tracks & Mixing Background Music...")
    temp_dubbed_video = merge_tts_and_mux_video(
        video_path=video_path,
        manifest_json=tts_manifest,
        background_audio=background_wav,
        output_video=output_video,
        bg_volume=bg_volume
    )
    dubbed_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "full_dubbed_audio.wav")
    if low_vram: clear_vram()

    # STEP 9: Lip-Syncing
    print(">>> STEP 9/9: Wav2Lip Lip Synchronization...")
    final_video = run_lip_sync(
        video_path=video_path,
        dubbed_audio_path=dubbed_audio_path,
        output_video=output_video,
        chunk_length_sec=chunk_sec,
        skip_lipsync=skip_lipsync,
        device_name=device
    )

    elapsed_sec = time.time() - start_time
    clear_vram()

    return {
        "video": video_path,
        "output_video": final_video,
        "subtitles": subtitle_files,
        "elapsed_sec": round(elapsed_sec, 2),
        "status": "SUCCESS"
    }


def run_batch_pipeline(
    batch_dir: str,
    output_dir: str = None,
    whisper_model: str = "small",
    demucs_model: str = "htdemucs",
    src_lang: str = "auto",
    tgt_lang: str = "auto",
    translation_engine: str = "auto",
    hf_token: str = None,
    device: str = "auto",
    low_vram: bool = True,
    skip_lipsync: bool = False
):
    """
    Processes all video files in a folder sequentially in Batch Mode, generating a summary report at the end.
    """
    if not os.path.exists(batch_dir):
        print(f"[ERROR] Batch directory not found: {batch_dir}", file=sys.stderr)
        sys.exit(1)

    video_extensions = ["*.mp4", "*.mkv", "*.mov", "*.avi"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(batch_dir, ext)))

    if not video_files:
        print(f"[ERROR] No video files found in batch directory: {batch_dir}", file=sys.stderr)
        sys.exit(1)

    batch_start_time = time.time()
    results = []

    print("\n" + "=" * 75)
    print(" 📦 ECHO LOCAL : BATCH PROCESSING PIPELINE INITIALIZED")
    print("=" * 75)
    print(f" Batch Folder : {batch_dir}")
    print(f" Total Videos : {len(video_files)}")
    print(f" Low VRAM Mode: ENABLED")
    print("=" * 75 + "\n")

    for idx, vid_path in enumerate(video_files, start=1):
        print(f"\n[BATCH PROGRESS {idx}/{len(video_files)}] Processing: {os.path.basename(vid_path)}")
        try:
            res = run_single_video_pipeline(
                video_path=vid_path,
                output_video=os.path.join(output_dir, f"{os.path.splitext(os.path.basename(vid_path))[0]}_dubbed.mp4") if output_dir else None,
                whisper_model=whisper_model,
                demucs_model=demucs_model,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                translation_engine=translation_engine,
                hf_token=hf_token,
                device=device,
                low_vram=low_vram,
                skip_lipsync=skip_lipsync
            )
            results.append(res)
        except Exception as e:
            print(f"[BATCH ERROR] Failed processing {vid_path}: {e}", file=sys.stderr)
            results.append({
                "video": vid_path,
                "error": str(e),
                "elapsed_sec": 0.0,
                "status": "FAILED"
            })
        finally:
            clear_vram()

    total_batch_time = time.time() - batch_start_time

    # Batch Summary Report
    successful_list = [r for r in results if r["status"] == "SUCCESS"]
    failed_list = [r for r in results if r["status"] == "FAILED"]

    print("\n" + "=" * 75)
    print(" 📊 ECHO LOCAL BATCH PROCESSING SUMMARY REPORT")
    print("=" * 75)
    print(f" Total Duration   : {total_batch_time / 60:.2f} minutes ({total_batch_time:.2f}s)")
    print(f" Total Processed  : {len(results)}")
    print(f" ✅ Successful     : {len(successful_list)}")
    print(f" ❌ Failed         : {len(failed_list)}")
    print("-" * 75)

    if successful_list:
        print("\n✅ SUCCESSFUL DUBBED VIDEOS:")
        for r in successful_list:
            print(f"  • {os.path.basename(r['video'])} -> {r['output_video']} ({r['elapsed_sec']}s)")

    if failed_list:
        print("\n❌ FAILED VIDEOS:")
        for r in failed_list:
            print(f"  • {os.path.basename(r['video'])}: {r['error']}")

    print("=" * 75 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="EchoLocal: AI Video Translation, Multi-Speaker Voice Cloning, Subtitle & Lip-Sync Pipeline."
    )
    parser.add_argument(
        "-i", "--video",
        default=None,
        help="Path to single input video file"
    )
    parser.add_argument(
        "-y", "--youtube-url",
        default=None,
        help="YouTube video URL to download and process (e.g. https://youtu.be/xxxxx)"
    )
    parser.add_argument(
        "-b", "--batch-dir",
        default=None,
        help="Path to folder containing multiple video files for Batch Mode processing"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output video file/folder path"
    )
    parser.add_argument(
        "-m", "--model",
        default="small",
        help="Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3'). Default: 'small'"
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name for vocal separation. Default: 'htdemucs'"
    )
    parser.add_argument(
        "--src-lang",
        default="auto",
        help="Source language ('hi', 'en', 'auto'). Default: auto"
    )
    parser.add_argument(
        "--tgt-lang",
        default="auto",
        help="Target language ('en', 'hi', 'auto'). Default: auto"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "indictrans2", "nllb"],
        default="auto",
        help="Translation engine. Default: auto"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote speaker diarization models"
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=True,
        help="Enable strict sequential execution and VRAM cleanup (Default: True)"
    )
    parser.add_argument(
        "--skip-lipsync",
        action="store_true",
        help="Skip Wav2Lip lip-synchronization step"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. Default: auto"
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum resolution for YouTube download (default: 1080p)"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # YouTube URL: download first, then process as a normal video
    # ------------------------------------------------------------------ #
    if args.youtube_url:
        print(f"\n[INFO] YouTube URL detected: {args.youtube_url}")
        try:
            video_path = download_youtube_video(
                url=args.youtube_url,
                max_height=args.max_height,
            )
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"[ERROR] YouTube download failed: {e}", file=sys.stderr)
            sys.exit(1)

        run_single_video_pipeline(
            video_path=video_path,
            output_video=args.output,
            whisper_model=args.model,
            demucs_model=args.demucs_model,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            translation_engine=args.engine,
            hf_token=args.hf_token,
            device=args.device,
            low_vram=args.low_vram,
            skip_lipsync=args.skip_lipsync
        )

    elif args.batch_dir:
        run_batch_pipeline(
            batch_dir=args.batch_dir,
            output_dir=args.output,
            whisper_model=args.model,
            demucs_model=args.demucs_model,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            translation_engine=args.engine,
            hf_token=args.hf_token,
            device=args.device,
            low_vram=args.low_vram,
            skip_lipsync=args.skip_lipsync
        )
    elif args.video:
        run_single_video_pipeline(
            video_path=args.video,
            output_video=args.output,
            whisper_model=args.model,
            demucs_model=args.demucs_model,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            translation_engine=args.engine,
            hf_token=args.hf_token,
            device=args.device,
            low_vram=args.low_vram,
            skip_lipsync=args.skip_lipsync
        )
    else:
        print(
            "[ERROR] Please specify one of:\n"
            "  --video <path>        Local video file\n"
            "  --youtube-url <url>   YouTube URL\n"
            "  --batch-dir <folder>  Folder of videos",
            file=sys.stderr
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
