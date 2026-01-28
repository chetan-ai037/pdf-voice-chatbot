import pyttsx3
_engine = None
def speak(text):
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
    _engine.say(text)
    _engine.runAndWait()