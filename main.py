import streamlit as st
import whisper
import os
import tempfile

def transcribe_video(video_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path)
    return result["segments"]

def save_srt(segments):
    srt_str = ""
    for i, segment in enumerate(segments, 1):
        start = format_time(segment["start"])
        end = format_time(segment["end"])
        text = segment["text"].strip()
        srt_str += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt_str

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Streamlit UI
st.title("🎬 Auto Subtitle Generator")
st.markdown("Upload a video file and generate subtitles automatically for it using OpenAI's Whisper.")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])
model_size = st.selectbox("Choose Whisper model size", ["tiny", "base", "small", "medium", "large"])

if st.button("Generate Subtitles") and video_file:
    with st.spinner("Processing..."):
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

        # Transcribe directly from video
        segments = transcribe_video(temp_video_path, model_size)

        # Generate .srt content
        srt_content = save_srt(segments)

        # Offer download
        st.success("✅ Subtitles generated!")
        st.download_button("Download .srt", srt_content, file_name="subtitles.srt", mime="text/plain")

        # Clean up
        os.remove(temp_video_path)
