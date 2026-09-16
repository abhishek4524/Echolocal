import argparse
import asyncio
import json
import os
import sys
import time
import shutil
import wave
import ffmpeg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import clear_vram, get_vram_info, handle_cuda_oom, ensure_ffmpeg_in_path

# Fix Windows console encoding to UTF-8 so Hindi/Unicode text prints without error
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["COQUI_TOS_AGREED"] = "1"
ensure_ffmpeg_in_path()

try:
    import torch
except ImportError:
    torch = None

try:
    from TTS.api import TTS
except ImportError:
    TTS = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

def get_wav_duration(wav_path: str) -> float:
    """Returns total duration of a WAV file in seconds."""
    try:
        with wave.open(wav_path, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0

def find_and_cache_clean_speaker_reference(
    segments: list,
    speaker_id: str,
    clean_vocals_wav: str,
    speakers_dir: str,
    min_duration: float = 6.0,
    max_duration: float = 10.0
) -> str:
    """
    Finds and caches a continuous or concatenated 6-10 second clean speech reference audio clip
    for speaker_id exclusively from Demucs clean vocals.wav.
    """
    cache_path = os.path.join(speakers_dir, f"{speaker_id}_clean_ref.wav")
    
    if os.path.exists(cache_path) and get_wav_duration(cache_path) >= min_duration:
        return cache_path

    if not os.path.exists(clean_vocals_wav):
        return clean_vocals_wav

    speaker_segments = [s for s in segments if s.get("speaker", "SPEAKER_00") == speaker_id]
    
    if not speaker_segments:
        shutil.copyfile(clean_vocals_wav, cache_path)
        return cache_path

    # Attempt 1: Single long continuous clean segment >= min_duration (6.0s)
    best_single_segment = None
    best_dur = 0.0
    
    for seg in speaker_segments:
        s_dur = seg.get("end", 0.0) - seg.get("start", 0.0)
        if s_dur >= min_duration:
            best_single_segment = seg
            best_dur = s_dur
            break
        elif s_dur > best_dur:
            best_single_segment = seg
            best_dur = s_dur

    if best_single_segment and best_dur >= min_duration:
        s_start = best_single_segment.get("start", 0.0)
        s_end = min(s_start + max_duration, best_single_segment.get("end", s_start + max_duration))
        try:
            (
                ffmpeg
                .input(clean_vocals_wav, ss=s_start, to=s_end)
                .output(cache_path, ar=24000, ac=1, loglevel="error")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            clip_dur = get_wav_duration(cache_path)
            print(f"[SPEAKER CACHE] Found continuous {clip_dur:.2f}s clean speech for {speaker_id} -> {cache_path}")
            return cache_path
        except Exception:
            pass

    # Attempt 2: Accumulate multiple speech segments for speaker_id until reaching 6-10s
    slice_dir = os.path.join(speakers_dir, f"temp_slices_{speaker_id}")
    os.makedirs(slice_dir, exist_ok=True)
    
    accumulated_files = []
    accumulated_dur = 0.0

    for idx, seg in enumerate(speaker_segments):
        s_start = seg.get("start", 0.0)
        s_end = seg.get("end", s_start + 1.0)
        seg_dur = s_end - s_start
        
        if seg_dur < 0.3:
            continue
            
        slice_path = os.path.join(slice_dir, f"slice_{idx:04d}.wav")
        try:
            (
                ffmpeg
                .input(clean_vocals_wav, ss=s_start, to=s_end)
                .output(slice_path, ar=24000, ac=1, loglevel="error")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            accumulated_files.append(slice_path)
            accumulated_dur += seg_dur
        except Exception:
            pass

        if accumulated_dur >= min_duration:
            break

    if accumulated_files:
        if len(accumulated_files) == 1:
            shutil.copyfile(accumulated_files[0], cache_path)
        else:
            concat_txt = os.path.join(slice_dir, "concat.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for sf in accumulated_files:
                    norm_p = os.path.abspath(sf).replace("\\", "/")
                    f.write(f"file '{norm_p}'\n")
            try:
                (
                    ffmpeg
                    .input(concat_txt, format="concat", safe=0)
                    .output(cache_path, c="copy", loglevel="error")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except Exception:
                shutil.copyfile(accumulated_files[0], cache_path)

        try:
            shutil.rmtree(slice_dir)
        except Exception:
            pass

        clip_len = get_wav_duration(cache_path)
        print(f"[SPEAKER CACHE] Accumulated {clip_len:.2f}s clean speech for {speaker_id} -> {cache_path}")
        return cache_path

    # Fallback to full clean_vocals_wav
    shutil.copyfile(clean_vocals_wav, cache_path)
    return cache_path


async def _edge_tts_synth(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def prepare_punctuation_for_tts(text: str, lang_code: str) -> str:
    """
    Punctuation-aware TTS pre-processing.
    Ensures natural phrasing, commas, and period pauses so speech cadence sounds human.
    """
    if not text:
        return text

    import re
    cleaned = text.strip()

    # Ensure spacing after punctuation (e.g. "hello,world" -> "hello, world")
    cleaned = re.sub(r'([,;:\?!।])([^\s0-9])', r'\1 \2', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if lang_code == "hi":
        # Insert subtle pause commas after key conversational connectors if not already present
        connectors = ["उदाहरण के लिए", "लेकिन", "इसलिए", "तो", "हालांकि", "वैसे", "दरअसल"]
        for conn in connectors:
            pattern = r'\b(' + re.escape(conn) + r')\b(?!\s*[,।\?\!])'
            cleaned = re.sub(pattern, r'\1,', cleaned)

        if not cleaned.endswith(("।", ".", "?", "!")):
            cleaned += "।"
    else:
        if not cleaned.endswith((".", "?", "!")):
            cleaned += "."

    return cleaned


def normalize_audio_loudness(input_wav: str, output_wav: str, target_i: float = -16.0):
    """
    Normalizes generated audio segment to a consistent target loudness (-16 LUFS) using
    FFmpeg loudnorm filter. Guarantees uniform volume level across all video segments.
    """
    try:
        (
            ffmpeg
            .input(input_wav)
            .filter('loudnorm', I=target_i, LRA=11, TP=-1.5)
            .output(output_wav, ar=24000, ac=1, loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        if input_wav != output_wav and os.path.exists(input_wav):
            shutil.copyfile(input_wav, output_wav)


def edge_tts_synthesize(text: str, lang_code: str, speaker_id: str, output_path: str):
    """Fallback neural TTS using Microsoft Edge-TTS with per-speaker voice mapping."""
    speaker_index = 0
    try:
        if "SPEAKER_" in speaker_id:
            speaker_index = int(speaker_id.replace("SPEAKER_", ""))
    except ValueError:
        speaker_index = 0

    hi_voices = ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
    en_voices = ["en-US-AndrewNeural", "en-US-GuyNeural", "en-US-BrianNeural", "en-US-AvaNeural", "en-US-EmmaNeural"]

    if lang_code.lower() == "hi":
        selected_voice = hi_voices[speaker_index % len(hi_voices)]
    else:
        selected_voice = en_voices[speaker_index % len(en_voices)]
    
    asyncio.run(_edge_tts_synth(text, selected_voice, output_path))
    
    if output_path.endswith(".wav"):
        temp_mp3 = output_path.replace(".wav", "_temp.mp3")
        if os.path.exists(output_path):
            os.rename(output_path, temp_mp3)
            (
                ffmpeg
                .input(temp_mp3)
                .output(output_path, ar=24000, ac=1, loglevel="error")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)


def generate_voice_cloned_tts(
    input_json: str,
    ref_audio: str,
    output_dir: str = None,
    language: str = "auto",
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    device_name: str = "auto",
    force_coqui: bool = False
) -> str:
    """
    Generates speaker-specific voice-cloned TTS audio segments using Coqui XTTS-v2 as primary engine.
    Extracts and caches clean 6-10s reference audio clips from Demucs vocals.wav per speaker.
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Transcript JSON not found: {input_json}")

    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

    device = "cuda" if (torch and torch.cuda.is_available() and device_name in ["cuda", "auto"]) else "cpu"

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    segments = data.get("segments", [])

    if not segments:
        print("[WARNING] No segments found in input JSON file.", file=sys.stderr)
        return ""

    if language == "auto" or not language:
        target_lang = metadata.get("target_language", "en").lower()
    else:
        target_lang = language.lower()

    lang_map = {
        "english": "en", "en": "en",
        "hindi": "hi", "hi": "hi",
        "spanish": "es", "es": "es",
        "french": "fr", "fr": "fr"
    }
    xtts_lang = lang_map.get(target_lang, "en")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "temp", "tts_segments")
    
    speakers_dir = os.path.join(project_root, "temp", "speakers")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(speakers_dir, exist_ok=True)

    use_coqui = TTS is not None
    if force_coqui and not use_coqui:
        print("[ERROR] --force-coqui enabled but Coqui TTS module is not installed.", file=sys.stderr)
        sys.exit(1)

    primary_engine_label = "Coqui XTTS-v2 (Zero-Shot Voice Cloning)" if use_coqui else "Edge-TTS Neural (Fallback)"

    print("=" * 75)
    print(f"[INFO] Input JSON        : {input_json}")
    print(f"[INFO] Primary TTS Engine: {primary_engine_label}")
    print(f"[INFO] Source Clean Vocals: {ref_audio} (Demucs Speech Track)")
    print(f"[INFO] Target Language   : {xtts_lang.upper()}")
    print(f"[INFO] Output Directory  : {output_dir}")
    print(f"[INFO] Compute Device    : {device.upper()}")
    print(f"[INFO] VRAM Status       : {get_vram_info()}")
    print("=" * 75)

    # 1. Extract & Cache 6-10s Clean Reference Audio Clips per Speaker (Demucs Vocals)
    speaker_ref_map = {}
    unique_speakers = sorted(list(set([seg.get("speaker", "SPEAKER_00") for seg in segments])))
    print(f"[INFO] Found {len(unique_speakers)} unique speakers: {unique_speakers}")

    for spk in unique_speakers:
        cached_clip = find_and_cache_clean_speaker_reference(
            segments=segments,
            speaker_id=spk,
            clean_vocals_wav=ref_audio,
            speakers_dir=speakers_dir,
            min_duration=6.0,
            max_duration=10.0
        )
        speaker_ref_map[spk] = cached_clip

    # 2. Initialize Primary Engine (Coqui XTTS-v2)
    tts_engine = None
    is_fallback = False

    if use_coqui:
        try:
            print(f"[INFO] Loading Coqui XTTS-v2 model ({model_name}) on {device.upper()}...")
            start_load = time.time()
            tts_engine = TTS(model_name=model_name).to(device)
            print(f"[INFO] Coqui XTTS-v2 loaded in {time.time() - start_load:.2f} seconds.")
        except Exception as e:
            print(f"\n[WARNING] Coqui XTTS-v2 load failed ({e}). Falling back to Edge-TTS...", file=sys.stderr)
            use_coqui = False
            is_fallback = True
    else:
        is_fallback = True
        print("\n[WARNING] Edge-TTS fallback used! Coqui XTTS-v2 was not loaded (Python 3.13 environment).", file=sys.stderr)

    manifest_segments = []
    total_segments = len(segments)
    start_total_time = time.time()

    print(f"\n[INFO] Synthesizing {total_segments} audio segments...")

    for idx, seg in enumerate(segments):
        text_to_speak = seg.get("translated_text") or seg.get("text", "")
        text_to_speak = text_to_speak.strip()

        if not text_to_speak:
            continue

        # 1. Punctuation-aware formatting for natural pause cadence
        text_to_speak = prepare_punctuation_for_tts(text_to_speak, xtts_lang)

        spk = seg.get("speaker", "SPEAKER_00")
        speaker_ref_wav = speaker_ref_map.get(spk, ref_audio)
        ref_dur = get_wav_duration(speaker_ref_wav)

        seg_filename = f"segment_{idx:04d}.wav"
        seg_output_path = os.path.join(output_dir, seg_filename)

        start_fmt = f"{int(seg['start'] // 60):02d}:{seg['start'] % 60:05.2f}"
        end_fmt = f"{int(seg['end'] // 60):02d}:{seg['end'] % 60:05.2f}"

        print(f"\n[SEGMENT {idx + 1}/{total_segments}] Time: [{start_fmt} -> {end_fmt}]")
        print(f"  - Speaker ID       : {spk}")
        print(f"  - Engine Active    : {'Coqui XTTS-v2 (Voice Clone)' if (use_coqui and tts_engine) else 'Edge-TTS Neural (Fallback)'}")
        if is_fallback:
            print(f"  - [WARNING] Edge-TTS fallback used! Coqui XTTS-v2 unavailable.")
        print(f"  - Ref Audio Clip   : {speaker_ref_wav} (Duration: {ref_dur:.2f}s - Clean Vocals)")
        print(f"  - Text to Speak    : \"{text_to_speak}\"")

        try:
            raw_seg_path = seg_output_path.replace(".wav", "_raw.wav")
            if use_coqui and tts_engine:
                tts_engine.tts_to_file(
                    text=text_to_speak,
                    speaker_wav=speaker_ref_wav,
                    language=xtts_lang,
                    file_path=raw_seg_path
                )
            else:
                edge_tts_synthesize(
                    text=text_to_speak,
                    lang_code=xtts_lang,
                    speaker_id=spk,
                    output_path=raw_seg_path
                )

            # 2. Loudness Normalization (-16 LUFS) for consistent volume level across segments
            normalize_audio_loudness(raw_seg_path, seg_output_path, target_i=-16.0)
            if os.path.exists(raw_seg_path):
                os.remove(raw_seg_path)

            seg_duration = get_wav_duration(seg_output_path)
            print(f"  - Generated Audio  : {seg_output_path} (Duration: {seg_duration:.2f}s - Normalized -16 LUFS)")

            manifest_segments.append({
                "segment_id": seg.get("id", idx),
                "speaker": spk,
                "engine_used": "Coqui XTTS-v2" if (use_coqui and tts_engine) else "Edge-TTS Neural (Fallback)",
                "is_fallback": is_fallback,
                "reference_clip": speaker_ref_wav,
                "reference_duration_sec": round(ref_dur, 2),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "target_duration": round(seg.get("end", 0.0) - seg.get("start", 0.0), 2),
                "generated_duration": round(seg_duration, 2),
                "original_text": seg.get("text", ""),
                "translated_text": text_to_speak,
                "audio_file": seg_output_path
            })

        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                handle_cuda_oom(e, f"TTS Segment {idx}", "Clearing VRAM...")
            else:
                print(f"[ERROR] Failed synthesizing segment {idx}: {e}", file=sys.stderr)

        clear_vram()

    total_duration = time.time() - start_total_time
    print("-" * 75)
    print(f"[INFO] Speech synthesis completed in {total_duration:.2f} seconds.")

    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "metadata": {
            "primary_engine": primary_engine_label,
            "fallback_triggered": is_fallback,
            "reference_audio_source": ref_audio,
            "target_language": xtts_lang,
            "speakers_detected": unique_speakers,
            "speaker_references": speaker_ref_map,
            "total_segments": len(manifest_segments),
            "total_processing_time_sec": round(total_duration, 2)
        },
        "segments": manifest_segments
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] TTS segment files & manifest saved to: {output_dir}")
    return manifest_path

def main():
    parser = argparse.ArgumentParser(
        description="Speaker-specific Voice-Clone / Neural TTS generation."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to translated transcript JSON file (from translate.py)"
    )
    parser.add_argument(
        "-r", "--ref-audio",
        required=True,
        help="Path to reference clean audio WAV file (from Demucs vocals.wav)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Optional output directory for segment WAV files"
    )
    parser.add_argument(
        "-l", "--language",
        default="auto",
        help="Target language code ('en', 'hi', 'auto'). Default: auto"
    )
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model name"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. Default: auto"
    )
    parser.add_argument(
        "--force-coqui",
        action="store_true",
        help="Force Coqui XTTS-v2 execution and error out if TTS module is unavailable"
    )

    args = parser.parse_args()
    generate_voice_cloned_tts(
        input_json=args.input,
        ref_audio=args.ref_audio,
        output_dir=args.output_dir,
        language=args.language,
        model_name=args.model,
        device_name=args.device,
        force_coqui=args.force_coqui
    )

if __name__ == "__main__":
    main()
