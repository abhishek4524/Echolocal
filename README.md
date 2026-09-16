# 🎙️ EchoLocal: Open-Source AI Video Dubbing, Multi-Speaker Voice Cloning & Lip-Sync Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**EchoLocal** is a production-grade, open-source AI video localization & dubbing pipeline. It automatically translates and dubs videos between **Hindi** and **English** (bi-directional), preserves original background music & sound effects (SFX), performs multi-speaker identification, generates timestamped `.srt` subtitles, and synchronizes lip movements.

Engineered for high performance, low VRAM consumption (optimized for 6GB GPUs like NVIDIA GeForce RTX 3050), and 100% crash-proof execution.

---

## 🌟 Key Features

- ▶️ **YouTube URL Integration**: Paste any YouTube video, Short, or Live stream URL (`yt-dlp`) for auto-download and dubbing at up to 1080p max resolution.
- ⚡ **GPU-Accelerated & Low VRAM**: PyTorch CUDA 12.4 acceleration with sequential model lifecycle management and VRAM auto-cleanup (runs smoothly on 6GB VRAM GPUs).
- 🎵 **Demucs Vocal & Background Separation**: Uses Meta's `htdemucs` to isolate speech from background music & SFX for crisp transcription and audio preservation.
- 🎙️ **Faster-Whisper STT**: High-precision speech-to-text recognition with CTranslate2 float16 GPU acceleration.
- 👥 **Hybrid Speaker Diarization**: Multi-speaker identification via Pyannote 3.1 + an automatic **Librosa MFCC + Pitch (F0) + KMeans Acoustic Feature Clustering** fallback (works out of the box without requiring a Hugging Face token!).
- 🌐 **Conversational Translation**: Integrated with `Helsinki-NLP` (`opus-mt-en-hi` & `opus-mt-hi-en`) & `NLLB-200` with an **Adaptation Post-Processing Pass** that refines literal machine translation into natural, colloquial conversational Hindi/English.
- 🗣️ **Coqui XTTS-v2 Zero-Shot Voice Cloning**: Clones speaker-specific voices directly on **Python 3.13** using 6–10 second clean reference audio clips extracted per speaker.
- 🎚️ **Loudness Normalization & 75ms Crossfading**: FFmpeg `loudnorm` filter (-16 LUFS) per segment + 75ms boundary crossfading to eliminate volume jumps, audio clipping, and multi-speaker overlap disturbance.
- 🎶 **Transparent Background Music Ducking**: Preserves original background music and SFX with dynamic sidechain ducking so background audio stays audible throughout the video.
- 📜 **Multi-Format Subtitles**: Auto-generates `.srt` subtitle files in **Original**, **Translated**, and **Bilingual (Dual-Language Stacked)** formats.
- 💋 **Wav2Lip Lip-Syncing**: VRAM-safe 10-second chunked lip synchronization with `--skip-lipsync` bypass mode.
- 💻 **Streamlit GUI & CLI Batch Processing**: Interactive Web UI (`app.py`) + single video & batch directory CLI commands (`run_pipeline.py`).

---

## 🔄 End-to-End Pipeline Architecture

```text
       ┌─────────────────────────────────────────────────────────┐
       │   Input Source: Local Video File OR YouTube Video URL   │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼ (Step 0: download_youtube.py / yt-dlp)
                      [input/<video_name>.mp4] (1080p max)
                                    │
                                    ▼ (Step 1: extract_audio.py)
                      [temp/<video_name>.wav] (16kHz Mono)
                                    │
                                    ▼ (Step 2: separate_audio.py - Demucs htdemucs)
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼ Vocals Track                                  ▼ Background Track (Music + SFX)
  [temp/separated/vocals.wav]                      [temp/separated/background.wav]
            │                                               │
            ▼ (Step 3: transcribe.py - faster-whisper GPU)  │
  [output/vocals.json]                                      │
            │                                               │
            ▼ (Step 4: diarize.py - Pyannote / Acoustic)    │
  [output/vocals_diarized.json (Multi-Speaker Tags)]        │
            │                                               │
            ▼ (Step 5: translate.py + Conversational Pass) │
  [output/vocals_diarized_translated.json]                │
            │                                               │
            ├───────────────────────────────────────────────┤
            │ (Step 6: generate_subtitles.py)               │
            ▼                                               │
  [.srt Subtitles: Original, Translated, Bilingual]         │
            │                                               │
            ▼ (Step 7: voice_clone_tts.py - Coqui XTTS-v2)  │
  [temp/tts_segments/segment_0000.wav (-16 LUFS)]          │
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼ (Step 8: merge_final.py)
              [Audio Speed-Sync + 75ms Crossfade + Demucs BG Ducking]
                                    │
                  [temp/full_dubbed_audio.wav]
                                    │
                                    ▼ (Step 9: lip_sync.py - Wav2Lip 10s Chunks)
                                    │
                    ┌───────────────┴───────────────┐
                    │      Final Output Video       │
                    │  (output/<video>_dubbed.mp4)  │
                    └───────────────────────────────┘
```

---

## 🛠️ Module Breakdown

| Module | File Path | Key Technology & Role |
|---|---|---|
| **YouTube Downloader** | [src/download_youtube.py](file:///F:/Echolocal/src/download_youtube.py) | Downloads YouTube videos/Shorts via `yt-dlp` at max 1080p resolution with safe filename sanitization and error handling. |
| **Audio Extractor** | [src/extract_audio.py](file:///F:/Echolocal/src/extract_audio.py) | Extracts 16kHz mono PCM WAV audio using FFmpeg. Auto-locates system FFmpeg. |
| **Audio Separator** | [src/separate_audio.py](file:///F:/Echolocal/src/separate_audio.py) | Separates audio into `vocals.wav` (speech) and `background.wav` (music & SFX) using Meta Demucs (`htdemucs`). |
| **Speech Transcriber** | [src/transcribe.py](file:///F:/Echolocal/src/transcribe.py) | Transcribes speech to text with timestamps using `faster-whisper` (CTranslate2 CUDA float16). |
| **Speaker Diarizer** | [src/diarize.py](file:///F:/Echolocal/src/diarize.py) | Diarizes multi-speaker turns (`SPEAKER_00`, `SPEAKER_01`) via Pyannote 3.1 with Librosa MFCC + Pitch KMeans fallback. |
| **Translator** | [src/translate.py](file:///F:/Echolocal/src/translate.py) | Translates Hindi↔English (Helsinki-NLP / NLLB-200) + Conversational Adaptation post-processing pass. |
| **Subtitle Generator**| [src/generate_subtitles.py](file:///F:/Echolocal/src/generate_subtitles.py) | Generates Original, Translated, and Bilingual stacked `.srt` subtitle files. |
| **Voice Clone TTS** | [src/voice_clone_tts.py](file:///F:/Echolocal/src/voice_clone_tts.py) | Zero-shot voice cloning with **Coqui XTTS-v2** + Edge-TTS fallback + Punctuation pre-processing + `-16 LUFS` loudnorm. |
| **Audio Synchronizer**| [src/merge_final.py](file:///F:/Echolocal/src/merge_final.py) | Syncs speech segment speeds (`atempo` 0.85x–1.30x), applies 75ms crossfades, mixes background track, and muxes video. |
| **Lip Synchronizer** | [src/lip_sync.py](file:///F:/Echolocal/src/lip_sync.py) | Synchronizes speaker lip movements with dubbed audio using Wav2Lip with 10-second chunking. |
| **GPU Manager** | [src/gpu_utils.py](file:///F:/Echolocal/src/gpu_utils.py) | Manages CUDA VRAM memory cleanup (`clear_vram()`), auto-detects FFmpeg binary, and handles OOM exceptions. |

---

## 💻 Setup & Installation

### 1. Requirements
- **Operating System**: Windows 10/11 or Linux
- **GPU**: NVIDIA GPU with 6GB+ VRAM (e.g., RTX 3050, RTX 3060, RTX 4060, T4)
- **Python**: 3.10, 3.11, or 3.13
- **FFmpeg**: Must be installed or available in `C:\ffmpeg\`

### 2. Installation Steps

```bash
# Clone repository
git clone https://github.com/your-username/EchoLocal.git
cd EchoLocal

# Create and activate virtual environment
python -m venv venv

# PowerShell (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies (with PyTorch CUDA 12.4 support)
pip install -r requirements.txt
```

### 3. Verify GPU Acceleration

```bash
python test_gpu.py
```
> **Expected Output**: `CUDA Available: True | GPU 0: NVIDIA GeForce RTX 3050 6GB Laptop GPU`

---

## 🚀 How to Run

### Option 1: Streamlit Web Interface (Recommended)

Launch the interactive Web GUI:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your web browser.
- Select between **📁 Local File Upload** or **▶️ YouTube URL**.
- Choose source/target languages and feature toggles (Voice Cloning, Background Music, Lip-Sync).
- Click **🚀 Start Dubbing Pipeline** to run the complete pipeline and preview/download results.

### Option 2: CLI — Process Local Video

```bash
python run_pipeline.py --video input/sample_video.mp4
```

### Option 3: CLI — Download & Process YouTube URL

```bash
python run_pipeline.py --youtube-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --max-height 1080
```

### Option 4: CLI — Fast Preview (Skip Lip-Sync)

```bash
python run_pipeline.py --video input/sample_video.mp4 --skip-lipsync
```

### Option 5: CLI — Batch Folder Processing

Place multiple videos in `input/` folder and process all at once:

```bash
python run_pipeline.py --batch-dir input/
```

---

## 📂 Project Directory Structure

```text
EchoLocal/
├── app.py                   # Streamlit Web Application
├── run_pipeline.py          # Master CLI Orchestrator (Single & Batch Mode)
├── test_gpu.py              # GPU Verification Script
├── requirements.txt         # Dependency Manifest
├── README.md                # Documentation
├── .streamlit/
│   └── config.toml          # Streamlit UI Configuration
├── input/                   # Input video files (.mp4, .mkv, .mov)
├── output/                  # Final dubbed videos, .SRT subtitles & JSON transcripts
├── temp/                    # Temporary audio cache, separated tracks & segment WAVs
└── src/
    ├── download_youtube.py  # YouTube Video Downloader (yt-dlp)
    ├── extract_audio.py     # Step 1: FFmpeg Audio Extractor
    ├── separate_audio.py    # Step 2: Demucs Vocals & Background Separator
    ├── transcribe.py        # Step 3: faster-whisper STT
    ├── diarize.py           # Step 4: Pyannote / Acoustic Feature Diarizer
    ├── translate.py         # Step 5: Helsinki-NLP / NLLB + Conversational Adaptation
    ├── generate_subtitles.py# Step 6: Subtitle Generator (.srt)
    ├── voice_clone_tts.py   # Step 7: Coqui XTTS-v2 Speech Generator
    ├── merge_final.py       # Step 8: Audio Speed Sync, Crossfade & Music Mixer
    ├── lip_sync.py          # Step 9: Wav2Lip Lip-Synchronizer
    └── gpu_utils.py         # GPU VRAM Tracker, clear_vram(), CUDA OOM Handler
```

---

## 📄 Output Artifacts

All final generated outputs are saved in `output/`:

- 🎥 **Final Dubbed Video**: `output/<video_name>_dubbed.mp4`
- 📜 **Original Subtitles**: `output/<video_name>_original.srt`
- 📜 **Translated Subtitles**: `output/<video_name>_translated.srt`
- 📜 **Bilingual Subtitles**: `output/<video_name>_bilingual.srt`
- 📄 **Transcript JSON**: `output/vocals_diarized_translated.json`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bug reports, feature suggestions, or translation model support for additional languages.

---

## 📜 License

This project is open-source under the **MIT License**. See the [LICENSE](LICENSE) file for details.
