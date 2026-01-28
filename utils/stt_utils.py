import whisper
import tempfile
from pydub import AudioSegment
import io
_model = None
def load_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model
def transcribe_audio(audio_bytes):
    model = load_model()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        audio_path = f.name
    result = model.transcribe(audio_path)
    text = result.get("text", "").strip()
    language = result.get("language", "en")
    return text, language