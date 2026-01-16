from setuptools import setup, find_packages

setup(
    name="pdf-voice-chatbot",
    version="1.0.0",
    description="Advanced Secure RAG Voice-enabled PDF Chatbot",
    author="Chetan Nag",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "pypdf",
        "sentence-transformers",
        "faiss-cpu",
        "langchain-text-splitters",
        "groq",
        "python-dotenv",
        "whisper",
        "pydub",
        "pyttsx3",
        "numpy",
        "ffmpeg-python"
    ],
    python_requires=">=3.10",
)
