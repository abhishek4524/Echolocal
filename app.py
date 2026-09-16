import os
import sys
import time
import streamlit as st

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gpu_utils import clear_vram, get_vram_info
from src.download_youtube import download_youtube_video, is_youtube_url
from src.extract_audio import extract_audio
from src.separate_audio import separate_vocals_and_background
from src.transcribe import transcribe_audio
from src.diarize import run_diarization
from src.translate import translate_transcript
from src.generate_subtitles import generate_subtitles_from_json
from src.voice_clone_tts import generate_voice_cloned_tts
from src.merge_final import merge_tts_and_mux_video
from src.lip_sync import run_lip_sync

# Streamlit Page Config
st.set_page_config(
    page_title="EchoLocal - AI Video Dubbing & Lip-Sync",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #4A90E2;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #357ABD;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎙️ EchoLocal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI Video Translation, Multi-Speaker Voice Cloning & Wav2Lip Lip-Sync</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Settings")

hf_token = st.sidebar.text_input(
    "Hugging Face Token (Pyannote Diarization)",
    type="password",
    help="Optional token for pyannote/speaker-diarization-3.1 model"
)

whisper_model = st.sidebar.selectbox(
    "Whisper Model Size",
    ["small", "tiny", "base", "medium", "large-v3"],
    index=0,
    help="'small' is optimal for RTX 3050 6GB VRAM"
)

translation_engine = st.sidebar.selectbox(
    "Translation Engine",
    ["auto", "indictrans2", "nllb"],
    index=0
)

bg_volume = st.sidebar.slider(
    "Background Music Volume",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05
)

low_vram_mode = st.sidebar.checkbox("Low VRAM Mode (Sequential + Memory Flush)", value=True)

st.sidebar.divider()
st.sidebar.markdown(f"**GPU Memory Status:**\n`{get_vram_info()}`")

# Main Interface Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Video or Enter YouTube URL")

    input_tab, yt_tab = st.tabs(["📁 Local File Upload", "▶️ YouTube URL"])

    input_video_path = None
    youtube_download_error = None

    with input_tab:
        uploaded_file = st.file_uploader(
            "Upload Video File (.mp4, .mkv, .mov)",
            type=["mp4", "mkv", "mov"]
        )
        if uploaded_file:
            input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
            os.makedirs(input_dir, exist_ok=True)
            input_video_path = os.path.join(input_dir, uploaded_file.name)
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"📁 File saved: `{uploaded_file.name}`")

    with yt_tab:
        yt_url_input = st.text_input(
            "Paste YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports youtube.com/watch, youtu.be, Shorts and Live URLs"
        )
        yt_max_height = st.selectbox(
            "Max Resolution",
            options=[1080, 720, 480],
            index=0,
            help="1080p recommended for best quality / reasonable file size"
        )

        if yt_url_input:
            if not is_youtube_url(yt_url_input):
                st.error("❌ This doesn't look like a valid YouTube URL. Please check and try again.")
            else:
                st.info(f"🔗 URL accepted: `{yt_url_input}`")

    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        src_lang_input = st.selectbox(
            "Source Language",
            ["Auto-Detect", "Hindi", "English"],
            index=0
        )
    with lang_col2:
        tgt_lang_input = st.selectbox(
            "Target Language",
            ["English", "Hindi"],
            index=0
        )

    st.subheader("2. Dubbing & Feature Controls")
    preserve_voice = st.checkbox("Preserve Speaker Voice (Multi-Speaker Voice Cloning)", value=True)
    preserve_bg_music = st.checkbox("Preserve Background Music & SFX (Demucs Separation)", value=True)
    enable_lipsync = st.checkbox("Enable Lip-Syncing (Wav2Lip)", value=True, help="Disable for faster processing without lip synchronization")

    start_button = st.button("🚀 Start Dubbing Pipeline")

with col2:
    st.subheader("3. Pipeline Progress & Results")

    # Language Code Mapping
    lang_code_map = {"Auto-Detect": "auto", "Hindi": "hi", "English": "en"}
    src_code = lang_code_map[src_lang_input]
    tgt_code = lang_code_map[tgt_lang_input]

    if start_button:
        # ---------------------------------------------------------------- #
        # Determine video source: YouTube URL or local file upload
        # ---------------------------------------------------------------- #
        active_yt_url = yt_url_input if yt_url_input and is_youtube_url(yt_url_input) else None
        has_local_file = input_video_path and os.path.exists(input_video_path)

        if not active_yt_url and not has_local_file:
            st.error("⚠️ Please upload a video file OR enter a valid YouTube URL first!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                clear_vram()

                # -------------------------------------------------------- #
                # YouTube download (if URL was provided)
                # -------------------------------------------------------- #
                if active_yt_url:
                    status_text.markdown("⏳ **Step 0:** Downloading YouTube video...")
                    progress_bar.progress(5)
                    with st.spinner(f"Downloading from YouTube ({yt_max_height}p max)..."):
                        try:
                            input_video_path = download_youtube_video(
                                url=active_yt_url,
                                max_height=yt_max_height,
                            )
                            st.success(f"✅ YouTube video downloaded: `{os.path.basename(input_video_path)}`")
                        except (ValueError, RuntimeError, ImportError) as dl_err:
                            st.error(f"❌ YouTube Download Failed: {dl_err}")
                            st.stop()

                filename_stem = os.path.splitext(os.path.basename(input_video_path))[0]
                st.info(f"📁 Processing: `{os.path.basename(input_video_path)}`")

                # STEP 1: Audio Extraction
                status_text.markdown("⏳ **Step 1/8:** Extracting Audio from Video...")
                progress_bar.progress(12)
                extracted_wav = extract_audio(input_file=input_video_path)
                if low_vram_mode: clear_vram()

                # STEP 2: Demucs Separation
                if preserve_bg_music:
                    status_text.markdown("⏳ **Step 2/8:** Separating Vocals & Background Track (Demucs)...")
                    progress_bar.progress(25)
                    vocals_wav, background_wav = separate_vocals_and_background(input_wav=extracted_wav)
                else:
                    vocals_wav = extracted_wav
                    background_wav = None
                if low_vram_mode: clear_vram()

                # STEP 3: Whisper Transcription
                status_text.markdown("⏳ **Step 3/8:** Transcribing Clean Vocals (faster-whisper)...")
                progress_bar.progress(37)
                transcript_json = transcribe_audio(
                    input_file=vocals_wav,
                    model_name=whisper_model,
                    language=src_code if src_code != "auto" else None
                )
                if low_vram_mode: clear_vram()

                # STEP 4: Pyannote Diarization
                status_text.markdown("⏳ **Step 4/8:** Identifying Speakers (Pyannote Diarization)...")
                progress_bar.progress(50)
                diarized_json = run_diarization(
                    audio_wav=vocals_wav,
                    transcript_json=transcript_json,
                    hf_token=hf_token
                )
                if low_vram_mode: clear_vram()

                # STEP 5: Translation
                status_text.markdown("⏳ **Step 5/9:** Translating Transcript (Helsinki-NLP/NLLB + Adaptation)...")
                progress_bar.progress(55)
                translated_json = translate_transcript(
                    input_file=diarized_json,
                    src_lang=src_code,
                    tgt_lang=tgt_code,
                    engine=translation_engine
                )
                if low_vram_mode: clear_vram()

                # STEP 6: Subtitle Generation
                status_text.markdown("⏳ **Step 6/9:** Generating .SRT Subtitle Files...")
                progress_bar.progress(65)
                subtitle_files = generate_subtitles_from_json(input_json=translated_json)

                # STEP 7: Multi-Speaker Voice Clone TTS
                status_text.markdown("⏳ **Step 7/9:** Generating Voice-Cloned Speech (Coqui XTTS-v2)...")
                progress_bar.progress(75)
                ref_audio_for_tts = vocals_wav if preserve_voice else extracted_wav
                tts_manifest = generate_voice_cloned_tts(
                    input_json=translated_json,
                    ref_audio=ref_audio_for_tts
                )
                if low_vram_mode: clear_vram()

                # STEP 7: Audio Sync & Video Muxing
                status_text.markdown("⏳ **Step 7/8:** Syncing Audio Segments & Mixing Tracks...")
                progress_bar.progress(87)
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                os.makedirs(output_dir, exist_ok=True)

                temp_dubbed_video = os.path.join(output_dir, f"{filename_stem}_dubbed.mp4")

                merge_tts_and_mux_video(
                    video_path=input_video_path,
                    manifest_json=tts_manifest,
                    background_audio=background_wav if preserve_bg_music else None,
                    output_video=temp_dubbed_video,
                    bg_volume=bg_volume
                )
                if low_vram_mode: clear_vram()

                # STEP 8: Wav2Lip Lip-Synchronization
                status_text.markdown("⏳ **Step 8/8:** Running Wav2Lip Lip Synchronization...")
                progress_bar.progress(100)

                final_video_path = os.path.join(output_dir, f"{filename_stem}_final_dubbed.mp4")
                dubbed_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "full_dubbed_audio.wav")

                run_lip_sync(
                    video_path=input_video_path,
                    dubbed_audio_path=dubbed_audio_path,
                    output_video=final_video_path,
                    skip_lipsync=not enable_lipsync
                )

                status_text.success("🎉 Pipeline Completed Successfully!")

                # Video Preview & Download
                st.subheader("📹 Final Dubbed Video Preview")
                st.video(final_video_path)

                with open(final_video_path, "rb") as video_file:
                    st.download_button(
                        label="📥 Download Final Dubbed Video",
                        data=video_file,
                        file_name=f"{filename_stem}_dubbed.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"❌ Pipeline Execution Error: {e}")
            finally:
                clear_vram()
