import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import clear_vram, get_vram_info, handle_cuda_oom

try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

def get_device_and_compute_type(requested_device: str):
    """Determine the optimal device and compute_type for faster-whisper."""
    if requested_device == "cuda":
        if torch and torch.cuda.is_available():
            return "cuda", "float16"
        else:
            return "cpu", "int8"
    elif requested_device == "cpu":
        return "cpu", "int8"
    else:  # auto
        if torch and torch.cuda.is_available():
            return "cuda", "float16"
        else:
            return "cpu", "int8"

def transcribe_audio(
    input_file: str,
    output_file: str = None,
    model_name: str = "small",
    device_name: str = "auto",
    language: str = None,
    beam_size: int = 5
) -> str:
    """
    Transcribe audio file using faster-whisper and save segments to JSON.
    Unloads model immediately after transcription to free VRAM.
    """
    if WhisperModel is None:
        print("[ERROR] 'faster-whisper' package is not installed. Please run pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    if output_file is None:
        filename_stem = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{filename_stem}.json")

    device, compute_type = get_device_and_compute_type(device_name)
    
    print("=" * 60)
    print(f"[INFO] Audio File    : {input_file}")
    print(f"[INFO] Model Size    : {model_name}")
    print(f"[INFO] Compute Device: {device.upper()} (Compute Type: {compute_type})")
    print(f"[INFO] VRAM Status   : {get_vram_info()}")
    print("=" * 60)

    model = None
    segments_list = []
    transcribe_duration = 0.0

    try:
        print(f"[INFO] Loading faster-whisper '{model_name}' model...")
        start_time = time.time()
        
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=models_dir
        )

        print(f"[INFO] Model loaded in {time.time() - start_time:.2f} seconds.")
        print("[INFO] Starting transcription...")
        
        transcribe_start = time.time()
        segments_generator, info = model.transcribe(
            input_file,
            beam_size=beam_size,
            language=language
        )

        print(f"[INFO] Detected Language: '{info.language}' (Probability: {info.language_probability:.2%})")
        print(f"[INFO] Audio Duration : {info.duration:.2f} seconds")
        print("-" * 60)

        for segment in segments_generator:
            start_fmt = f"{int(segment.start // 60):02d}:{segment.start % 60:05.2f}"
            end_fmt = f"{int(segment.end // 60):02d}:{segment.end % 60:05.2f}"
            text_clean = segment.text.strip()
            
            print(f"[{start_fmt} -> {end_fmt}] {text_clean}")
            
            segments_list.append({
                "id": segment.id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": text_clean,
                "avg_logprob": round(segment.avg_logprob, 4),
                "no_speech_prob": round(segment.no_speech_prob, 4)
            })

        transcribe_duration = time.time() - transcribe_start
        print("-" * 60)
        print(f"[INFO] Transcription completed in {transcribe_duration:.2f} seconds.")

        result_data = {
            "metadata": {
                "audio_file": os.path.basename(input_file),
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
                "model": model_name,
                "device": device,
                "compute_type": compute_type,
                "transcription_time_sec": round(transcribe_duration, 2)
            },
            "segments": segments_list
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Transcript saved to: {output_file}")

    except Exception as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            handle_cuda_oom(e, "Whisper Transcription", "Try using a smaller model size (--model tiny/base) or --device cpu.")
        else:
            print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        # VRAM Cleanup
        clear_vram(model)
        print(f"[INFO] VRAM Cleaned: {get_vram_info()}")

    return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe WAV audio file using faster-whisper (CTranslate2) with GPU acceleration."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input WAV audio file (e.g., temp/sample.wav)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional custom output JSON path. Defaults to output/<audio_filename>.json"
    )
    parser.add_argument(
        "-m", "--model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large-v3). Default: 'small'"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run inference on. Default: 'auto'"
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code (e.g., 'hi', 'en'). Default: Auto-detect"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding. Default: 5"
    )

    args = parser.parse_args()
    transcribe_audio(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        device_name=args.device,
        language=args.language,
        beam_size=args.beam_size
    )

if __name__ == "__main__":
    main()
