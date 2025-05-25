import streamlit as st
import whisper
import os
import tempfile
import logging
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Check if FFmpeg is available, try to install if not"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.warning("⚠️ FFmpeg not found. Attempting to install...")
        try:
            # Try to install ffmpeg-python as a fallback
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'])
            st.info("✅ Installed ffmpeg-python")
            return True
        except:
            st.error("❌ Could not install FFmpeg. Please contact support.")
            return False

def transcribe_video(video_path, model_size="base"):
    """Transcribe video using Whisper model with error handling"""
    try:
        logger.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        
        logger.info(f"Transcribing video: {video_path}")
        result = model.transcribe(video_path)
        
        return result["segments"]
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise e

def save_srt(segments):
    """Convert segments to SRT format"""
    srt_str = ""
    for i, segment in enumerate(segments, 1):
        start = format_time(segment["start"])
        end = format_time(segment["end"])
        text = segment["text"].strip()
        srt_str += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt_str

def format_time(seconds):
    """Format seconds to SRT time format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# Streamlit UI
st.title("🎬 Auto Subtitle Generator")
st.markdown("Upload a video file and generate subtitles automatically using OpenAI's Whisper.")

# Add file size warning
st.warning("⚠️ Large video files may take several minutes to process and consume significant memory.")

video_file = st.file_uploader(
    "Upload a video file", 
    type=["mp4", "mkv", "avi", "mov", "webm", "flv"],
    help="Supported formats: MP4, MKV, AVI, MOV, WebM, FLV"
)

model_size = st.selectbox(
    "Choose Whisper model size", 
    ["tiny", "base", "small", "medium", "large"],
    help="Larger models are more accurate but slower and use more memory"
)

if video_file:
    # Show file info
    st.info(f"File: {video_file.name} ({video_file.size / (1024*1024):.1f} MB)")
    
    if video_file.size > 100 * 1024 * 1024:  # 100MB
        st.warning("⚠️ Large file detected. This may take a while to process.")

# Check FFmpeg before allowing processing
ffmpeg_ok = check_ffmpeg()

if st.button("Generate Subtitles") and video_file and ffmpeg_ok:
    temp_video_path = None
    try:
        with st.spinner("Processing video... This may take several minutes."):
            # Create progress placeholder
            progress_text = st.empty()
            
            progress_text.text("📁 Saving uploaded file...")
            
            # Save uploaded video temporarily with proper extension
            file_extension = os.path.splitext(video_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_video:
                temp_video.write(video_file.read())
                temp_video_path = temp_video.name
            
            progress_text.text("🤖 Loading Whisper model...")
            
            # Transcribe video
            segments = transcribe_video(temp_video_path, model_size)
            
            progress_text.text("📝 Generating subtitles...")
            
            # Generate .srt content
            srt_content = save_srt(segments)
            
            progress_text.text("✅ Complete!")
            
            # Show success and download
            st.success("✅ Subtitles generated successfully!")
            
            # Show preview of first few subtitles
            preview_lines = srt_content.split('\n\n')[:3]
            st.text_area("Preview (first 3 subtitles):", '\n\n'.join(preview_lines), height=150)
            
            # Download button
            st.download_button(
                "📥 Download .srt file", 
                srt_content, 
                file_name=f"{os.path.splitext(video_file.name)[0]}_subtitles.srt", 
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        logger.error(f"Processing error: {str(e)}")
        
        # Common error suggestions
        if "out of memory" in str(e).lower():
            st.error("💾 Memory error: Try using a smaller model (tiny/base) or a shorter video.")
        elif "format" in str(e).lower() or "codec" in str(e).lower():
            st.error("🎥 Video format error: Try converting your video to MP4 format.")
        elif "permission" in str(e).lower():
            st.error("🔒 Permission error: Check file permissions and disk space.")
        else:
            st.error("🔧 Try using a different model size or check that your video file is valid.")
            
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

# Add footer with requirements
st.markdown("---")
st.markdown("**Requirements:** `pip install streamlit openai-whisper`")
st.markdown("**Note:** First run will download the selected Whisper model (~39MB for 'tiny', ~244MB for 'base')")