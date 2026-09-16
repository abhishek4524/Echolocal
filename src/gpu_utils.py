import gc
import os
import shutil
import sys

try:
    import torch
except ImportError:
    torch = None

def ensure_ffmpeg_in_path() -> bool:
    """
    Locates ffmpeg.exe in common Windows directories and automatically
    adds its folder to os.environ['PATH'].
    """
    if shutil.which("ffmpeg") is not None:
        return True

    common_paths = [
        "C:\\ffmpeg\\bin",
        "C:\\ffmpeg",
        "C:\\Program Files\\ffmpeg\\bin",
        "C:\\ProgramData\\chocolatey\\bin"
    ]

    # Search inside C:\ffmpeg subfolders if extracted
    if os.path.exists("C:\\ffmpeg"):
        for root, dirs, files in os.walk("C:\\ffmpeg"):
            if "ffmpeg.exe" in files:
                common_paths.append(root)

    for candidate in common_paths:
        if os.path.exists(os.path.join(candidate, "ffmpeg.exe")):
            os.environ["PATH"] = candidate + os.path.pathsep + os.environ.get("PATH", "")
            print(f"[INFO] Auto-detected FFmpeg at: {candidate}")
            return True

    return False

def get_vram_info() -> str:
    """Returns formatted string of current GPU memory usage."""
    if torch and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return f"Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB | Peak: {max_allocated:.1f} MB"
    return "GPU Memory Info: CUDA not active"

def clear_vram(*models):
    """
    Deletes model objects, forces garbage collection,
    and clears CUDA cache to release VRAM immediately.
    """
    for model in models:
        if model is not None:
            try:
                del model
            except Exception:
                pass

    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def handle_cuda_oom(error: Exception, step_name: str, fallback_hint: str = None):
    """Handles CUDA Out of Memory errors gracefully."""
    print(f"\n[CRITICAL ERROR] CUDA Out of Memory (OOM) encountered during: {step_name}", file=sys.stderr)
    print(f"Details: {error}", file=sys.stderr)
    if fallback_hint:
        print(f"[HINT] {fallback_hint}", file=sys.stderr)
    print("[INFO] Attempting VRAM cleanup...", file=sys.stderr)
    clear_vram()
