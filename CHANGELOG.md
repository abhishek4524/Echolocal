# Changelog

All notable changes to EchoLocal are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) | Versioning: [SemVer](https://semver.org/)

---

## [1.1.0] — 2024-xx-xx  *(Upcoming)*

### Added
- `api/server.py` — FastAPI REST API with async job queue
  - `POST /api/v1/dub` — submit dubbing job (file upload or YouTube URL)
  - `GET /api/v1/status/{id}` — poll job progress
  - `GET /api/v1/download/{id}` — stream finished video
  - `GET /api/v1/jobs` — list all jobs
  - `DELETE /api/v1/jobs/{id}` — delete job & artifacts
- `src/config.py` — centralized configuration via frozen `dataclasses`
- `src/logger.py` — structured logging module (replaces `print()` throughout)
- `tests/` — pytest unit test suite (runs without GPU in CI)
  - `test_translate.py` — adaptation & unicode robustness tests
  - `test_download.py` — filename sanitization & URL validation tests
  - `test_voice_clone_tts.py` — punctuation & speed-factor tests
  - `test_config.py` — config defaults & immutability tests
  - `test_diarize.py` — speaker label assignment tests
  - `conftest.py` — shared fixtures (sample segments, dummy audio/video)
- `benchmark.py` — CLI benchmarking tool with per-step VRAM + RTF metrics
- `Dockerfile` — GPU-enabled container (`pytorch/pytorch:2.6.0-cuda12.4`)
- `docker-compose.yml` — multi-service compose (API + Streamlit)
- `.github/workflows/ci.yml` — GitHub Actions CI (lint, typecheck, test, docker build)
- `CONTRIBUTING.md` — developer guide with PR checklist & architecture diagram

### Changed
- All pipeline modules updated with full type hints
- Logging migrated from `print()` to `logging.Logger` throughout `src/`
- Magic numbers centralized in `src/config.py`

---

## [1.0.0] — 2024-01-15

### Added
- End-to-end AI video dubbing pipeline (local, no cloud API required)
- **YouTube URL support** via `yt-dlp` (max 1080p, safe filename sanitization)
- **Multi-speaker diarization** — Pyannote Audio with MFCC + Pitch KMeans acoustic fallback
- **Voice cloning TTS** — Coqui XTTS-v2 (multilingual, per-speaker voice matching)
  - Python 3.13 compatibility patches (coqpit `is_union`, `torch.load weights_only`)
  - Punctuation-aware generation (natural comma/period pauses)
  - Per-segment FFmpeg `loudnorm` normalization (−16 LUFS target)
  - 75ms crossfade between adjacent TTS segments
- **Conversational translation adaptation** — post-processing pass after Helsinki-NLP MT
  - Fixes common Hindi literal translation artifacts (`आप को` → `आपको`, etc.)
- **Multi-speaker audio mixing** — timeline boundary enforcement + max-envelope blending
  - Eliminates inter-speaker overlap disturbance
- **Background music preservation** — gentle sidechain ducking during speech segments
- **Streamlit GUI** (`app.py`) — local file upload + YouTube URL input
- **CLI orchestrator** (`run_pipeline.py`) — single video, batch, and YouTube modes
- **Batch processing** — process entire folders automatically
- **GPU auto-detection** — CUDA 12.4 + PyTorch 2.6, CPU fallback
- **Lip sync** — Wav2Lip chunked inference for 6 GB VRAM constraint
- **Subtitle generation** — `.srt` output alongside dubbed video
- **Audio separation** — Demucs vocals/accompaniment stem splitting
- `.gitignore`, `LICENSE` (MIT), `README.md` with badges

### Technical Highlights
- RTX 3050 6 GB Laptop GPU optimized (sequential model loading, `int8` quantization)
- ~3× real-time TTS synthesis speed on GPU
- `sys.stdout.reconfigure(encoding='utf-8')` for Hindi console output on Windows

---

## Legend

| Prefix | Meaning |
|---|---|
| **Added** | New features |
| **Changed** | Changes to existing functionality |
| **Deprecated** | Features to be removed in a future release |
| **Removed** | Removed features |
| **Fixed** | Bug fixes |
| **Security** | Vulnerability patches |
