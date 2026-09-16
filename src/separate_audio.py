import argparse
import os
import sys
import shutil
import subprocess
import time
import wave
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import clear_vram, get_vram_info, handle_cuda_oom

try:
    import torch
except ImportError:
    torch = None

def separate_vocals_and_background(
    input_wav: str,
    output_dir: str = None,
    model_name: str = "htdemucs",
    device_name: str = "auto"
) -> tuple[str, str]:
    """
    Separates audio file into vocals (speech) and background (music + SFX) using Demucs.
    """
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Input audio file not found: {input_wav}")

    device = "cuda" if (torch and torch.cuda.is_available() and device_name in ["cuda", "auto"]) else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    filename_stem = os.path.splitext(os.path.basename(input_wav))[0]
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "temp", "separated", filename_stem)
    
    os.makedirs(output_dir, exist_ok=True)

    vocals_path = os.path.join(output_dir, "vocals.wav")
    background_path = os.path.join(output_dir, "background.wav")

    print("=" * 60)
    print(f"[INFO] Input Audio Path: {input_wav}")
    print(f"[INFO] Demucs Model   : {model_name}")
    print(f"[INFO] Compute Device : {device.upper()}")
    print(f"[INFO] VRAM Status    : {get_vram_info()}")
    print("=" * 60)

    print(f"[INFO] Running Demucs audio separation...")
    start_time = time.time()

    success = False
    separator = None

    try:
        from demucs.api import Separator, save_audio
        
        separator = Separator(model=model_name, device=device, shifts=1)
        origin, stems = separator.separate_audio_file(input_wav)
        
        vocals_tensor = stems.get("vocals")
        bg_tensor = origin - vocals_tensor if vocals_tensor is not None else origin
        
        save_audio(vocals_tensor, vocals_path, samplerate=separator.samplerate)
        save_audio(bg_tensor, background_path, samplerate=separator.samplerate)
        
        success = True
        print("[INFO] Audio separation via Demucs API succeeded.")

    except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else Exception) as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            handle_cuda_oom(e, "Demucs Vocal Separation", "Retrying separation on CPU...")
            device = "cpu"
        else:
            print(f"[WARNING] Demucs API separation failed ({e}). Retrying via CLI...")

    finally:
        # VRAM Cleanup
        clear_vram(separator)
        print(f"[INFO] VRAM Cleaned: {get_vram_info()}")

    # Fallback CLI execution if API failed
    if not success:
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", model_name,
            "--two-stems", "vocals",
            "-d", device,
            "-o", os.path.join(project_root, "temp", "demucs_out"),
            input_wav
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            demucs_track_dir = os.path.join(project_root, "temp", "demucs_out", model_name, filename_stem)
            
            cli_vocals = os.path.join(demucs_track_dir, "vocals.wav")
            cli_no_vocals = os.path.join(demucs_track_dir, "no_vocals.wav")
            
            if os.path.exists(cli_vocals):
                shutil.copyfile(cli_vocals, vocals_path)
            if os.path.exists(cli_no_vocals):
                shutil.copyfile(cli_no_vocals, background_path)
                
            success = True
        except Exception as e:
            print(f"[ERROR] Demucs CLI separation failed: {e}", file=sys.stderr)
            print("[WARNING] Fallback: Using original audio as vocals & background.")
            shutil.copyfile(input_wav, vocals_path)
            shutil.copyfile(input_wav, background_path)
        finally:
            clear_vram()

    sep_duration = time.time() - start_time
    print("-" * 60)
    print(f"[INFO] Separation completed in {sep_duration:.2f} seconds.")
    print(f"[SUCCESS] Vocals Track    : {vocals_path}")
    print(f"[SUCCESS] Background Track: {background_path}")
    print("-" * 60)

    return vocals_path, background_path

def main():
    parser = argparse.ArgumentParser(
        description="Separate audio into vocals and background (music+SFX) using Demucs (htdemucs)."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input audio WAV file (e.g. temp/sample.wav)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Optional output directory for separated tracks. Defaults to temp/separated/<filename_stem>/"
    )
    parser.add_argument(
        "-m", "--model",
        default="htdemucs",
        help="Demucs model name (default: 'htdemucs')"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. Default: auto"
    )

    args = parser.parse_args()
    separate_vocals_and_background(
        input_wav=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        device_name=args.device
    )

if __name__ == "__main__":
    main()
