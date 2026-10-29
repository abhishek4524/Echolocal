"""
benchmark.py
EchoLocal Pipeline Benchmarking Tool

Measures per-step latency, VRAM usage, and real-time factor for a given video.
Results are printed as a table and saved to benchmark_results.json.

Usage:
    python benchmark.py --input path/to/video.mp4 --target-lang hi
    python benchmark.py --input path/to/video.mp4 --steps transcribe,translate
    python benchmark.py --help
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Add src/ to path ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    step: str
    duration_s: float
    vram_before_mb: float
    vram_after_mb: float
    vram_peak_mb: float
    real_time_factor: Optional[float] = None  # step_time / video_duration
    extra: Dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    video_path: str
    video_duration_s: float
    target_lang: str
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    timestamp: str
    steps: List[StepResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    overall_real_time_factor: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# VRAM helpers
# ═════════════════════════════════════════════════════════════════════════════

def _vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except ImportError:
        pass
    return 0.0


def _vram_reserved_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**2
    except ImportError:
        pass
    return 0.0


def _reset_vram_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _vram_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except ImportError:
        pass
    return 0.0


def _gc_collect():
    """Force Python GC + CUDA cache clear between steps."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Step runner
# ═════════════════════════════════════════════════════════════════════════════

def run_step(name: str, fn: Callable, video_duration: float) -> StepResult:
    """Execute a pipeline step and capture timing + VRAM metrics."""
    _gc_collect()
    _reset_vram_peak()

    vram_before = _vram_mb()
    logger.info(f"▶  [{name}] starting  (VRAM: {vram_before:.0f} MB)")

    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0

    vram_after = _vram_mb()
    vram_peak = _vram_peak_mb()
    rtf = elapsed / video_duration if video_duration > 0 else None

    sr = StepResult(
        step=name,
        duration_s=round(elapsed, 3),
        vram_before_mb=round(vram_before, 1),
        vram_after_mb=round(vram_after, 1),
        vram_peak_mb=round(vram_peak, 1),
        real_time_factor=round(rtf, 3) if rtf is not None else None,
        extra=result if isinstance(result, dict) else {},
    )

    logger.info(
        f"✔  [{name}] {elapsed:.2f}s | VRAM peak: {vram_peak:.0f} MB | RTF: {rtf:.2f}x"
        if rtf else f"✔  [{name}] {elapsed:.2f}s | VRAM peak: {vram_peak:.0f} MB"
    )
    return sr


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark suite
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    video_path: Path,
    target_lang: str = "hi",
    steps_filter: Optional[List[str]] = None,
) -> BenchmarkReport:
    import torch
    import subprocess
    import platform

    # ── Get video duration ────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True
        )
        video_duration = float(result.stdout.strip())
    except Exception:
        logger.warning("Could not determine video duration — assuming 60s for RTF")
        video_duration = 60.0

    logger.info(f"Video: {video_path.name} | Duration: {video_duration:.1f}s | Target: {target_lang}")

    # ── System info ───────────────────────────────────────────────────────────
    gpu_name = None
    cuda_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda

    report = BenchmarkReport(
        video_path=str(video_path),
        video_duration_s=round(video_duration, 2),
        target_lang=target_lang,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        timestamp=datetime.utcnow().isoformat(),
    )

    # ── Temp workspace ────────────────────────────────────────────────────────
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="echolocal_bench_"))
    audio_path = tmp / "audio.wav"
    stems: Dict[str, str] = {}
    segments: list = []
    translated: list = []

    def should_run(step: str) -> bool:
        return steps_filter is None or step in steps_filter

    # ─ Step 1: Extract audio ──────────────────────────────────────────────────
    if should_run("extract_audio"):
        from extract_audio import extract_audio
        report.steps.append(run_step(
            "extract_audio",
            lambda: extract_audio(str(video_path), str(audio_path)) or {},
            video_duration,
        ))

    # ─ Step 2: Separate audio ─────────────────────────────────────────────────
    if should_run("separate_audio") and audio_path.exists():
        from separate_audio import separate_audio
        def _sep():
            nonlocal stems
            stems = separate_audio(str(audio_path), str(tmp))
            return {"stems": list(stems.keys())}
        report.steps.append(run_step("separate_audio", _sep, video_duration))

    # ─ Step 3: Transcribe ─────────────────────────────────────────────────────
    vocals = stems.get("vocals", str(audio_path))
    if should_run("transcribe") and Path(vocals).exists():
        from transcribe import transcribe_audio
        def _transcribe():
            nonlocal segments
            segments = transcribe_audio(vocals)
            return {"n_segments": len(segments)}
        report.steps.append(run_step("transcribe", _transcribe, video_duration))

    # ─ Step 4: Diarize ───────────────────────────────────────────────────────
    if should_run("diarize") and segments and Path(vocals).exists():
        from diarize import diarize_audio
        def _diarize():
            nonlocal segments
            segments = diarize_audio(vocals, segments)
            speakers = {s.get("speaker", "?") for s in segments}
            return {"n_speakers": len(speakers)}
        report.steps.append(run_step("diarize", _diarize, video_duration))

    # ─ Step 5: Translate ──────────────────────────────────────────────────────
    if should_run("translate") and segments:
        from translate import translate_segments
        def _translate():
            nonlocal translated
            translated = translate_segments(segments, target_lang=target_lang)
            return {"n_segments": len(translated)}
        report.steps.append(run_step("translate", _translate, video_duration))

    # ─ Step 6: TTS ───────────────────────────────────────────────────────────
    if should_run("tts") and translated:
        from voice_clone_tts import generate_dubbed_audio
        def _tts():
            generate_dubbed_audio(translated, vocals, str(tmp))
            return {}
        report.steps.append(run_step("tts", _tts, video_duration))

    # ─ Totals ─────────────────────────────────────────────────────────────────
    report.total_duration_s = round(sum(s.duration_s for s in report.steps), 3)
    report.overall_real_time_factor = round(
        report.total_duration_s / video_duration, 3
    ) if video_duration > 0 else 0.0

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Pretty-print report
# ═════════════════════════════════════════════════════════════════════════════

def print_report(report: BenchmarkReport) -> None:
    print("\n" + "═" * 72)
    print(f"  EchoLocal Benchmark Report — {report.timestamp}")
    print("═" * 72)
    print(f"  Video      : {Path(report.video_path).name}  ({report.video_duration_s:.1f}s)")
    print(f"  Target lang: {report.target_lang}")
    print(f"  Python     : {report.python_version}")
    print(f"  PyTorch    : {report.torch_version}")
    print(f"  CUDA       : {report.cuda_version or 'N/A'}")
    print(f"  GPU        : {report.gpu_name or 'CPU only'}")
    print("─" * 72)
    print(f"  {'Step':<22} {'Time (s)':>9} {'VRAM Peak':>11} {'RTF':>7}")
    print("─" * 72)
    for s in report.steps:
        rtf_str = f"{s.real_time_factor:.2f}x" if s.real_time_factor else "  —"
        print(f"  {s.step:<22} {s.duration_s:>9.3f} {s.vram_peak_mb:>9.0f} MB {rtf_str:>7}")
    print("─" * 72)
    print(f"  {'TOTAL':<22} {report.total_duration_s:>9.3f} {'':>11} {report.overall_real_time_factor:.2f}x")
    print("═" * 72)
    print()


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoLocal Pipeline Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video")
    parser.add_argument("--target-lang", "-t", default="hi", help="Target language code (default: hi)")
    parser.add_argument(
        "--steps", "-s",
        default=None,
        help="Comma-separated list of steps to run (e.g. transcribe,translate,tts). Default: all",
    )
    parser.add_argument("--output", "-o", default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose debug logging")
    args = parser.parse_args()

    if args.verbose:
        from logger import set_global_log_level
        set_global_log_level("DEBUG")

    video = Path(args.input)
    if not video.exists():
        print(f"[ERROR] File not found: {video}", file=sys.stderr)
        sys.exit(1)

    steps = [s.strip() for s in args.steps.split(",")] if args.steps else None

    report = run_benchmark(video, target_lang=args.target_lang, steps_filter=steps)
    print_report(report)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
