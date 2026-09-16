import argparse
import json
import os
import sys

def format_timestamp_srt(seconds: float) -> str:
    """
    Converts seconds (float) into SRT timestamp format: HH:MM:SS,mmm
    Example: 83.250 -> 00:01:23,250
    """
    seconds = max(0.0, seconds)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        seconds += 1.0
        millis = 0
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_subtitles_from_json(
    input_json: str,
    output_dir: str = None
) -> dict[str, str]:
    """
    Reads transcript/translation JSON file and generates SRT subtitle files:
    - Original language SRT (<filename>_original.srt)
    - Translated language SRT (<filename>_translated.srt)
    - Dual/Bilingual SRT (<filename>_bilingual.srt)
    """
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON file not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[WARNING] No segments found in JSON file for subtitle generation.", file=sys.stderr)
        return {}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    filename_stem = os.path.splitext(os.path.basename(input_json))[0]
    filename_stem = filename_stem.replace("_translated", "").replace("_diarized", "")
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    orig_srt_path = os.path.join(output_dir, f"{filename_stem}_original.srt")
    trans_srt_path = os.path.join(output_dir, f"{filename_stem}_translated.srt")
    bilingual_srt_path = os.path.join(output_dir, f"{filename_stem}_bilingual.srt")

    has_translation = any("translated_text" in seg for seg in segments)

    orig_lines = []
    trans_lines = []
    bilingual_lines = []

    for idx, seg in enumerate(segments, start=1):
        start_str = format_timestamp_srt(seg.get("start", 0.0))
        end_str = format_timestamp_srt(seg.get("end", seg.get("start", 0.0) + 1.0))
        time_header = f"{start_str} --> {end_str}"
        
        speaker = seg.get("speaker")
        spk_prefix = f"[{speaker}] " if speaker else ""
        
        text_orig = seg.get("text", "").strip()
        text_trans = seg.get("translated_text", "").strip() if has_translation else ""

        # Original SRT Block
        orig_lines.append(f"{idx}\n{time_header}\n{spk_prefix}{text_orig}\n\n")

        # Translated SRT Block
        if has_translation:
            trans_lines.append(f"{idx}\n{time_header}\n{spk_prefix}{text_trans}\n\n")
            bilingual_lines.append(f"{idx}\n{time_header}\n{spk_prefix}{text_orig}\n{text_trans}\n\n")

    # Write Files
    generated_files = {}

    with open(orig_srt_path, "w", encoding="utf-8") as f:
        f.writelines(orig_lines)
    generated_files["original"] = orig_srt_path
    print(f"[SUCCESS] Original SRT created: {orig_srt_path}")

    if has_translation:
        with open(trans_srt_path, "w", encoding="utf-8") as f:
            f.writelines(trans_lines)
        generated_files["translated"] = trans_srt_path
        print(f"[SUCCESS] Translated SRT created: {trans_srt_path}")

        with open(bilingual_srt_path, "w", encoding="utf-8") as f:
            f.writelines(bilingual_lines)
        generated_files["bilingual"] = bilingual_srt_path
        print(f"[SUCCESS] Bilingual SRT created: {bilingual_srt_path}")

    return generated_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate .srt subtitle files (Original, Translated, Bilingual) from EchoLocal transcript JSON."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input transcript or translated JSON file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Optional output directory for .srt files. Defaults to output/"
    )

    args = parser.parse_args()
    generate_subtitles_from_json(
        input_json=args.input,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
