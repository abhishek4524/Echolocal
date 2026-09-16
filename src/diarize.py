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

def get_hf_token(cli_token: str = None) -> str:
    """Retrieve Hugging Face token from CLI arg or environment variables."""
    if cli_token:
        return cli_token
    return (
        os.getenv("HF_TOKEN") or
        os.getenv("HUGGINGFACE_TOKEN") or
        os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

def assign_speakers_to_transcript(transcript_json: str, diarization_turns: list, output_json: str = None) -> str:
    """
    Merges pyannote diarization speaker turns with Whisper transcript segments
    by matching maximum temporal overlap.
    """
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[WARNING] No segments found in transcript JSON.", file=sys.stderr)
        return transcript_json

    last_speaker = "SPEAKER_00"

    for seg in segments:
        t_start = seg["start"]
        t_end = seg["end"]
        
        speaker_overlaps = {}
        for turn in diarization_turns:
            d_start, d_end, speaker = turn["start"], turn["end"], turn["speaker"]
            
            overlap = max(0.0, min(t_end, d_end) - max(t_start, d_start))
            if overlap > 0.0:
                speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0.0) + overlap

        if speaker_overlaps:
            best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            seg["speaker"] = best_speaker
            last_speaker = best_speaker
        else:
            seg["speaker"] = last_speaker

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    if output_json is None:
        filename_stem = os.path.splitext(os.path.basename(transcript_json))[0]
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_json = os.path.join(output_dir, f"{filename_stem}_diarized.json")

    data["metadata"]["diarization_enabled"] = True

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Diarized transcript saved to: {output_json}")
    return output_json


def run_acoustic_speaker_clustering(audio_wav: str, transcript_json: str, output_json: str = None) -> str:
    """
    Acoustic audio feature clustering fallback (Librosa MFCC + pitch + KMeans) for multi-speaker identification.
    Runs automatically when Pyannote Hugging Face Token is not provided or fails,
    ensuring multi-speaker voice cloning works out of the box!
    """
    print("[INFO] Running Acoustic Feature Speaker Clustering (Librosa MFCC + Pitch + KMeans)...")

    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return assign_speakers_to_transcript(transcript_json, [], output_json)

    try:
        import librosa
        import numpy as np
        from sklearn.cluster import KMeans

        y, sr = librosa.load(audio_wav, sr=16000)
        total_samples = len(y)

        features_list = []
        valid_indices = []

        for idx, seg in enumerate(segments):
            start_s = max(0.0, seg.get("start", 0.0))
            end_s = min(total_samples / sr, seg.get("end", start_s + 1.0))

            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            segment_audio = y[start_idx:end_idx]

            if len(segment_audio) < int(0.3 * sr):
                continue

            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # Pitch estimate (F0)
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
            pitch_vals = pitches[magnitudes > np.median(magnitudes)]
            mean_pitch = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0.0

            feat = np.hstack([mfcc_mean, mfcc_std, [mean_pitch * 0.01]])
            features_list.append(feat)
            valid_indices.append(idx)

        if len(features_list) >= 2:
            X = np.array(features_list)
            # Standardize feature matrix
            mean_x = np.mean(X, axis=0)
            std_x = np.std(X, axis=0) + 1e-6
            X_norm = (X - mean_x) / std_x

            # Determine number of clusters (2 speakers for standard dialogue)
            n_clusters = min(2, len(features_list))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_norm)

            for val_idx, label in zip(valid_indices, labels):
                segments[val_idx]["speaker"] = f"SPEAKER_{label:02d}"

            print(f"[SUCCESS] Acoustic speaker clustering completed: Identified {n_clusters} distinct speakers across {len(valid_indices)} segments.")
        else:
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"

    except Exception as e:
        print(f"[WARNING] Acoustic clustering failed ({e}). Defaulting to SPEAKER_00.")
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if output_json is None:
        filename_stem = os.path.splitext(os.path.basename(transcript_json))[0]
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_json = os.path.join(output_dir, f"{filename_stem}_diarized.json")

    data["metadata"]["diarization_enabled"] = True
    data["metadata"]["diarization_method"] = "Acoustic KMeans Clustering (Librosa MFCC)"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Diarized transcript saved to: {output_json}")
    return output_json


def run_diarization(
    audio_wav: str,
    transcript_json: str,
    output_json: str = None,
    hf_token: str = None,
    device_name: str = "auto"
) -> str:
    """
    Runs pyannote.audio speaker diarization on audio_wav (or acoustic feature clustering fallback)
    and assigns speaker tags to transcript_json.
    """
    if not os.path.exists(audio_wav):
        raise FileNotFoundError(f"Audio file not found: {audio_wav}")

    if not os.path.exists(transcript_json):
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_json}")

    token = get_hf_token(hf_token)
    device = "cuda" if (torch and torch.cuda.is_available() and device_name in ["cuda", "auto"]) else "cpu"

    print("=" * 60)
    print(f"[INFO] Audio File      : {audio_wav}")
    print(f"[INFO] Transcript JSON : {transcript_json}")
    print(f"[INFO] Compute Device  : {device.upper()}")
    print(f"[INFO] VRAM Status     : {get_vram_info()}")
    print("=" * 60)

    if not token:
        print("[NOTICE] Hugging Face Token (HF_TOKEN) is not set.")
        print("[FALLBACK] Switching to Acoustic Feature Speaker Clustering (MFCC + Pitch KMeans)...")
        return run_acoustic_speaker_clustering(audio_wav, transcript_json, output_json)

    pipeline = None
    diarization_turns = []

    try:
        from pyannote.audio import Pipeline
        
        print("[INFO] Loading pyannote/speaker-diarization-3.1 pipeline...")
        start_time = time.time()
        
        # Use token=token instead of use_auth_token for compatibility with newer pyannote/huggingface_hub
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        
        if torch and device == "cuda":
            pipeline.to(torch.device("cuda"))

        print(f"[INFO] Running pyannote speaker diarization...")
        diarization = pipeline(audio_wav)

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_turns.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker
            })
            print(f"  [{turn.start:05.2f}s -> {turn.end:05.2f}s] {speaker}")

        print(f"[INFO] Diarization finished in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            handle_cuda_oom(e, "Pyannote Diarization", "Retrying with acoustic clustering fallback...")
        else:
            print(f"[WARNING] Pyannote diarization failed ({e}). Falling back to acoustic clustering.", file=sys.stderr)
        return run_acoustic_speaker_clustering(audio_wav, transcript_json, output_json)

    finally:
        clear_vram(pipeline)
        print(f"[INFO] VRAM Cleaned: {get_vram_info()}")

    return assign_speakers_to_transcript(transcript_json, diarization_turns, output_json)


def main():
    parser = argparse.ArgumentParser(
        description="Speaker Diarization using pyannote.audio for multi-speaker identification."
    )
    parser.add_argument(
        "-a", "--audio",
        required=True,
        help="Path to input audio WAV file (e.g. temp/separated/.../vocals.wav or temp/sample.wav)"
    )
    parser.add_argument(
        "-t", "--transcript",
        required=True,
        help="Path to transcript JSON file (from transcribe.py)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output JSON path. Defaults to output/<file>_diarized.json"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face User Access Token for pyannote gated models"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. Default: auto"
    )

    args = parser.parse_args()
    run_diarization(
        audio_wav=args.audio,
        transcript_json=args.transcript,
        output_json=args.output,
        hf_token=args.hf_token,
        device_name=args.device
    )

if __name__ == "__main__":
    main()
