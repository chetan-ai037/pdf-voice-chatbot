import pyttsx3

_engine = None

def speak(text):
    global _engine
    try:
        if _engine is None:
            _engine = pyttsx3.init()
            _engine.setProperty("rate", 165)

        _engine.say(text)
        _engine.runAndWait()
    except Exception as e:
        print(f"Error in TTS: {e}")

def save_to_file(text, filename):
    global _engine
    try:
        if _engine is None:
            _engine = pyttsx3.init()
            _engine.setProperty("rate", 165)
            
        _engine.save_to_file(text, filename)
        _engine.runAndWait()
        return True
    except Exception as e:
        print(f"Error in TTS save: {e}")
        return False