# Contributing to EchoLocal

Thank you for considering contributing to **EchoLocal**! 🎙️  
This guide will help you get set up, understand the codebase, and submit quality contributions.

---

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Running Tests](#running-tests)
- [Coding Standards](#coding-standards)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).  
Be respectful, inclusive, and constructive.

---

## Getting Started

### Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.11, 3.12, 3.13 |
| CUDA | 11.8+ | Required for GPU pipeline; CPU fallback available |
| FFmpeg | 5.x+ | Must be in system `PATH` |
| Git | 2.x+ | |

### Fork & Clone

```bash
# 1. Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/Echolocal.git
cd Echolocal

# 2. Add upstream remote
git remote add upstream https://github.com/abhishek4524/Echolocal.git
```

---

## Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install dev-only extras
pip install pytest pytest-mock pytest-cov ruff pyright

# Verify GPU setup
python test_gpu.py
```

### Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `./models` | HuggingFace model cache directory |
| `TORCH_HOME` | `./models` | PyTorch model cache directory |
| `ECHOLOCAL_LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG / INFO / WARNING / ERROR) |

---

## Project Architecture

```
Echolocal/
├── src/                  # Core pipeline modules
│   ├── config.py         # Centralized config (dataclasses)
│   ├── logger.py         # Centralized logging
│   ├── download_youtube.py
│   ├── extract_audio.py
│   ├── separate_audio.py
│   ├── transcribe.py
│   ├── diarize.py
│   ├── translate.py
│   ├── generate_subtitles.py
│   ├── voice_clone_tts.py
│   ├── merge_final.py
│   ├── lip_sync.py
│   └── gpu_utils.py
├── api/                  # FastAPI REST server
│   └── server.py
├── tests/                # pytest unit & integration tests
│   ├── conftest.py       # Shared fixtures
│   ├── test_config.py
│   ├── test_translate.py
│   ├── test_download.py
│   ├── test_voice_clone_tts.py
│   └── test_diarize.py
├── app.py                # Streamlit GUI
├── run_pipeline.py       # CLI orchestrator
├── benchmark.py          # Performance benchmarking
├── Dockerfile
└── docker-compose.yml
```

### Pipeline Data Flow

```
Input Video
    │
    ▼
extract_audio  →  .wav
    │
    ▼
separate_audio (Demucs)  →  vocals.wav + no_vocals.wav
    │
    ▼
transcribe (faster-whisper)  →  segments [start, end, text]
    │
    ▼
diarize (Pyannote / MFCC KMeans)  →  segments + speaker labels
    │
    ▼
translate (Helsinki-NLP + adaptation)  →  translated segments
    │
    ▼
generate_subtitles  →  .srt file
    │
    ▼
voice_clone_tts (XTTS-v2)  →  per-speaker dubbed audio
    │
    ▼
merge_final (FFmpeg + sidechain duck)  →  full dubbed audio
    │
    ▼
lip_sync (Wav2Lip)  →  lip-synced video
    │
    ▼
Output Video (_dubbed.mp4)
```

---

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_translate.py -v

# Run only fast tests (skip heavy integration tests)
pytest tests/ -v -m "not slow"

# Run with verbose output
pytest tests/ -v -s
```

> [!NOTE]
> Tests are designed to run **without GPU or model files** — all heavy dependencies
> are stubbed with `unittest.mock`. No download required for CI.

---

## Coding Standards

### Style

- **Formatter**: [`ruff format`](https://docs.astral.sh/ruff/formatter/) (Black-compatible)
- **Linter**: [`ruff check`](https://docs.astral.sh/ruff/) — rules: E, F, W, I
- **Line length**: 100 characters max
- **Type hints**: Required for all new public functions

```bash
# Check & auto-fix style issues
ruff check src/ --fix
ruff format src/
```

### Logging

Use the centralized logger — **never use `print()` in pipeline code**:

```python
from logger import get_logger
logger = get_logger(__name__)

logger.info("Processing segment 3/10")
logger.warning("Low VRAM — switching to CPU fallback")
logger.error("FFmpeg failed: %s", exc)
```

### Configuration

**Never hardcode magic numbers**. Add to `src/config.py`:

```python
# ✅ Good
from config import cfg
threshold = cfg.audio.bg_duck_threshold

# ❌ Bad
threshold = 0.25
```

### Type Hints

All new functions must have complete type annotations:

```python
# ✅ Good
def sanitize_filename(title: str, max_length: int = 200) -> str:
    ...

# ❌ Bad
def sanitize_filename(title, max_length=200):
    ...
```

---

## Submitting a Pull Request

### Before Opening a PR

- [ ] All new code has type hints
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Linting passes: `ruff check src/ api/ tests/`
- [ ] Formatting passes: `ruff format --check src/ api/ tests/`
- [ ] New features have corresponding unit tests
- [ ] `CHANGELOG.md` updated (if applicable)
- [ ] No new `print()` statements in pipeline code (use logger)
- [ ] No hardcoded magic numbers (use `config.py`)

### PR Title Format

```
[feat]  Add real-time streaming TTS output
[fix]   Correct VRAM leak in separate_audio step
[docs]  Update Docker deployment guide
[test]  Add integration test for translate → tts pipeline
[perf]  Reduce XTTS model load time with lazy initialization
[ci]    Add matrix test for Python 3.13
```

### PR Description Template

```markdown
## What does this PR do?
Brief description of the change.

## Why is this needed?
Context / motivation.

## Testing
- [ ] Unit tests added / updated
- [ ] Tested locally with GPU
- [ ] Tested with sample video (duration: __ s)

## Screenshots / Benchmark Results (if applicable)
```

---

## Reporting Issues

When filing a bug, please include:

1. **OS** + Python version + GPU model
2. **Steps to reproduce** (exact command run)
3. **Full error traceback** (from `logs/echolocal.log` if available)
4. **Input video details** (duration, format, number of speakers)

Use the **Bug Report** template when creating an issue on GitHub.

---

## Areas Where Contributions Are Welcome

| Area | Description | Difficulty |
|---|---|---|
| 🌐 Languages | Add new translation language pairs | Easy |
| 🎨 Streamlit UI | Improve progress visualization | Medium |
| ⚡ Performance | Async pipeline stages / parallel TTS | Hard |
| 🐳 Docker | CPU-only Docker image | Medium |
| 📊 Benchmarks | Add more benchmark scenarios | Easy |
| 📝 Docs | Improve API docs / tutorials | Easy |
| 🧪 Tests | Add more unit & integration tests | Medium |
| 🔊 Audio | Better background music handling | Hard |

---

*Happy dubbing! 🎙️✨*
