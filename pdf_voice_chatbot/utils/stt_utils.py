import whisper
import tempfile
from pydub import AudioSegment
from pydub.effects import normalize
import io
import os

_model = None

def load_model():
    global _model
    if _model is None:
        try:
            # Using base model for better accuracy while maintaining speed
            _model = whisper.load_model("base")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
    return _model

def transcribe_audio(audio_bytes):
    """
    Transcribe audio bytes to text with improved audio processing.
    Returns: (transcribed_text, language)
    """
    try:
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Audio data is empty")
        
        model = load_model()
        
        # Handle audio conversion - try multiple formats
        audio = None
        formats_to_try = [
            None,  # Auto-detect
            "webm",
            "wav", 
            "ogg",
            "mp3",
            "m4a"
        ]
        
        for fmt in formats_to_try:
            try:
                if fmt:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                else:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                break
            except:
                continue
        
        if audio is None:
            raise ValueError("Unsupported audio format - could not read audio file")
        
        # Improved audio preprocessing
        # Normalize volume for better recognition
        audio = normalize(audio)
        
        # Remove silence at the beginning and end
        if len(audio) > 100:  # Only if audio is long enough
            audio = audio.strip_silence(silence_len=100, silence_thresh=-40)
        
        # Convert to WAV for Whisper with optimal settings
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            try:
                # Set optimal sample rate and channels for Whisper
                audio = audio.set_frame_rate(16000).set_channels(1)
                # Export with optimal settings
                audio.export(f.name, format="wav", parameters=["-acodec", "pcm_s16le"])
                path = f.name
            except Exception as e:
                # Clean up temp file
                if os.path.exists(f.name):
                    try:
                        os.unlink(f.name)
                    except:
                        pass
                raise ValueError(f"Failed to convert audio: {str(e)}")

        try:
            # Optimized transcription with better settings
            result = model.transcribe(
                path, 
                language="en",
                task="transcribe",
                fp16=False,  # More compatible
                verbose=False,
                condition_on_previous_text=False  # Faster, no context dependency
            )
            transcribed_text = result.get("text", "").strip()
            language = result.get("language", "en")
        finally:
            # Clean up temp file
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        
        # Clean up transcribed text - remove common artifacts
        if transcribed_text:
            # Remove leading/trailing whitespace and common transcription artifacts
            transcribed_text = transcribed_text.strip()
            # Remove "thank you for watching" or similar YouTube-style endings if present
            transcribed_text = transcribed_text.split("thank you")[0].strip()
            # Remove trailing punctuation artifacts
            while transcribed_text.endswith("..") or transcribed_text.endswith(",,"):
                transcribed_text = transcribed_text[:-1].strip()
        
        if not transcribed_text:
            raise ValueError("No speech detected in audio")
        
        return transcribed_text, language
        
    except Exception as e:
        raise RuntimeError(f"Audio transcription failed: {str(e)}")