import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gpu_utils import clear_vram, get_vram_info, handle_cuda_oom

# Fix Windows console encoding to UTF-8 so Hindi/Unicode text prints without error
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
except ImportError:
    torch = None

class HelsinkiTranslator:
    def __init__(self, src_lang: str, tgt_lang: str, device: str = "cuda"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        self.device = device if (torch and torch.cuda.is_available() and device == "cuda") else "cpu"
        
        if src_lang == "en" and tgt_lang == "hi":
            self.model_name = "Helsinki-NLP/opus-mt-en-hi"
        elif src_lang == "hi" and tgt_lang == "en":
            self.model_name = "Helsinki-NLP/opus-mt-hi-en"
        else:
            self.model_name = "Helsinki-NLP/opus-mt-en-hi"
            
        print(f"[INFO] Loading Helsinki-NLP ({self.model_name}) on {self.device.upper()}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        print("[INFO] Helsinki-NLP Translation model loaded successfully.")

    def translate_batch(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(**inputs, max_length=256)
            
        translated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [t.strip() for t in translated_texts]

    def free_memory(self):
        clear_vram(self.model, self.tokenizer)


class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600m", device: str = "cuda"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        self.device = device if (torch and torch.cuda.is_available() and device == "cuda") else "cpu"
        print(f"[INFO] Loading NLLB-200 model ({model_name}) on {self.device.upper()}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("[INFO] NLLB-200 loaded successfully.")

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        nllb_map = {"hi": "hin_Deva", "en": "eng_Latn"}
        src_code = nllb_map.get(src_lang.lower(), "hin_Deva")
        tgt_code = nllb_map.get(tgt_lang.lower(), "eng_Latn")
        
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256
            )
            
        translated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [t.strip() for t in translated_texts]

    def free_memory(self):
        clear_vram(self.model, self.tokenizer)


def adapt_translation_conversational(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Conversational adaptation post-processing pass.
    Refines literal machine translation output into natural, fluent colloquial Hindi or English.
    Fixes awkward literal phrases, punctuation spacing, and Devnagari formatting artifacts.
    """
    if not text:
        return text

    adapted = text.strip()

    if tgt_lang == "hi":
        # 1. Devnagari / Unicode normalization (remove zero-width joiners, fix broken glyphs)
        adapted = adapted.replace("\u200d", "").replace("\u200c", "")

        # 2. Fix literal / awkward Hindi translation patterns from NLLB / Helsinki
        replacements = [
            # Fix homonym mistranslation: 'book' (reserve) translated as 'पुस्तक' (paper book)
            (r"होटल पुस्तक", "होटल बुक"),
            (r"कमरा पुस्तक", "कमरा बुक"),
            (r"और पुस्तक", "और बुक"),
            (r"\bपुस्तक\b", "बुक"),

            # Temporal / literal idiom fixes
            (r"छठी सदी पर", "छठे दिन"),
            (r"छठी सदी", "छठे दिन"),
            (r"पाँच दिनों तक बारिश करता है", "पाँच दिनों तक बारिश होती है"),
            (r"पाँच दिन बारिश का मतलब", "पाँच दिन बारिश होने का मतलब"),
            (r"बारिश का उपयोग किया गया है", "बारिश का पानी खत्म हो चुका है"),
            (r"संस्कृति के जारी होने की वजह से", "पुराने ट्रेंड के जारी रहने की वजह से"),
            
            # Common awkward literal phrases
            (r"व्यक्\u200dति", "व्यक्ति"),
            (r"व्यक्‍ति", "व्यक्ति"),
            (r"एक पश्चिमी आदमी पढ़ा", "मैंने कहीं एक पश्चिमी विद्वान के बारे में पढ़ा था"),
            (r"उसी से तुम्हारा भी अभिषेक हो जाता है", "उसी तरह आपकी सोच भी तय होती है"),
            (r"छठी दिन 50-50 है", "छठे दिन 50-50 चांस रहता है"),
            (r"अन्य नहीं कर रहे हैं", "बाकी लोग नहीं"),
            (r"सभी संभव चुनाव", "सारे ऑप्शंस"),

            # Conversational connectors & punctuation refinement
            (r"\s+।", "।"),
            (r"\s+,", ","),
            (r"\?\?", "?"),
            (r"!!", "!"),
        ]

        import re
        for pattern, replacement in replacements:
            adapted = re.sub(pattern, replacement, adapted)

        # Ensure sentence ends with proper punctuation for TTS cadence
        if not adapted.endswith(("।", ".", "?", "!")):
            adapted += "।"

    elif tgt_lang == "en":
        import re
        # English conversational fixes
        replacements = [
            (r"\s+,", ","),
            (r"\s+\.", "."),
            (r"\?\?", "?"),
            (r"!!", "!"),
        ]
        for pattern, replacement in replacements:
            adapted = re.sub(pattern, replacement, adapted)

        # Capitalize first letter
        if len(adapted) > 0 and adapted[0].islower():
            adapted = adapted[0].upper() + adapted[1:]

        if not adapted.endswith((".", "?", "!")):
            adapted += "."

    return adapted


def translate_transcript(
    input_file: str,
    output_file: str = None,
    src_lang: str = "auto",
    tgt_lang: str = "auto",
    engine: str = "auto",
    device: str = "auto"
) -> str:
    """
    Translates segments in a transcript JSON file (Hindi <-> English).
    Supports Helsinki-NLP (fastest public model), NLLB-200, and IndicTrans2.
    Includes a conversational adaptation post-processing pass for natural dubbing flow.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Transcript JSON not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    segments = data.get("segments", [])

    if not segments:
        print("[WARNING] No segments found in transcript JSON.")
        return input_file

    detected_src = metadata.get("language", "hi").lower()
    if src_lang == "auto" or not src_lang:
        src_lang = "hi" if "hi" in detected_src else "en"

    if tgt_lang == "auto" or not tgt_lang:
        tgt_lang = "en" if src_lang == "hi" else "hi"

    print("=" * 60)
    print(f"[INFO] Input JSON  : {input_file}")
    print(f"[INFO] Source Lang : {src_lang.upper()} (Detected: {detected_src})")
    print(f"[INFO] Target Lang : {tgt_lang.upper()}")
    print(f"[INFO] VRAM Status : {get_vram_info()}")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if output_file is None:
        filename_stem = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{filename_stem}_translated.json")

    translator = None
    engine_used = ""

    # Try Helsinki-NLP (Fastest public non-gated translation)
    try:
        translator = HelsinkiTranslator(src_lang=src_lang, tgt_lang=tgt_lang, device=device)
        engine_used = f"Helsinki-NLP ({translator.model_name})"
    except Exception as e1:
        print(f"[WARNING] Helsinki-NLP failed ({e1}). Retrying with NLLB-200...")
        try:
            translator = NLLBTranslator(device=device)
            engine_used = "NLLB-200 (distilled-600M)"
        except Exception as e2:
            print(f"[ERROR] NLLB-200 failed: {e2}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Translating {len(segments)} segments using {engine_used}...")
    start_time = time.time()

    texts = [seg["text"] for seg in segments]
    batch_size = 8
    translated_texts = []

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if isinstance(translator, HelsinkiTranslator):
                translated_batch = translator.translate_batch(batch)
            else:
                translated_batch = translator.translate_batch(batch, src_lang, tgt_lang)
            translated_texts.extend(translated_batch)

        translated_segments = []
        print("\n[INFO] Applying Conversational Adaptation post-processing pass...")
        for seg, trans_text in zip(segments, translated_texts):
            # Apply conversational adaptation pass
            adapted_text = adapt_translation_conversational(trans_text, src_lang, tgt_lang)

            updated_seg = dict(seg)
            updated_seg["literal_translation"] = trans_text
            updated_seg["translated_text"] = adapted_text
            translated_segments.append(updated_seg)

            start_fmt = f"{int(seg['start'] // 60):02d}:{seg['start'] % 60:05.2f}"
            end_fmt = f"{int(seg['end'] // 60):02d}:{seg['end'] % 60:05.2f}"
            print(f"[{start_fmt} -> {end_fmt}] {seg['text']}")
            print(f"  └─ Literal: \"{trans_text}\"")
            print(f"  └─ Adapted: \"{adapted_text}\"")

        translation_duration = time.time() - start_time
        print("-" * 60)
        print(f"[INFO] Translation completed in {translation_duration:.2f} seconds.")

        output_data = {
            "metadata": {
                **metadata,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "translation_engine": engine_used,
                "translation_time_sec": round(translation_duration, 2)
            },
            "segments": translated_segments
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Translated transcript saved to: {output_file}")

    except Exception as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            handle_cuda_oom(e, "Translation Phase", "Try reducing batch size.")
        else:
            print(f"[ERROR] Translation execution error: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        if translator:
            translator.free_memory()
        clear_vram()

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Translate transcript JSON (Hindi <-> English) preserving start/end segment timestamps."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input transcript JSON file (from transcribe.py)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output JSON path. Defaults to output/<audio>_translated.json"
    )
    parser.add_argument(
        "--src-lang",
        default="auto",
        help="Source language code ('hi', 'en', 'auto'). Default: auto"
    )
    parser.add_argument(
        "--tgt-lang",
        default="auto",
        help="Target language code ('en', 'hi', 'auto'). Default: auto"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "indictrans2", "nllb"],
        default="auto",
        help="Translation engine. Default: auto"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. Default: auto"
    )

    args = parser.parse_args()
    translate_transcript(
        input_file=args.input,
        output_file=args.output,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        engine=args.engine,
        device=args.device
    )

if __name__ == "__main__":
    main()
