# PDF Voice Chatbot

This project is a voice-enabled chatbot that allows users to upload a PDF file and ask questions using either text or voice input.

The chatbot processes the uploaded document and generates answers strictly based on the content of the PDF.

---

## Project Description

The PDF Voice Chatbot enables users to interact with PDF documents in a conversational way.  
Users can upload a document, ask questions related to it, and receive accurate responses generated using a retrieval-based approach.

The system ensures that answers are derived only from the uploaded document and not from external sources.

---

## Features

- Upload PDF documents
- Ask questions using text input
- Ask questions using voice input
- Automatic speech-to-text conversion
- Document-based answer generation
- Optional text-to-speech output
- Secure retrieval-based response generation

---

## Working of the System

1. The user uploads a PDF document
2. Text is extracted from the PDF
3. The text is divided into smaller chunks
4. Each chunk is converted into vector embeddings
5. The user provides a question using text or voice
6. Relevant document chunks are retrieved using similarity search
7. A language model generates an answer based on retrieved content
8. The answer is displayed and can be converted to speech

---

## Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers (MiniLM)
- Groq LLM API
- Whisper (Speech-to-Text)
- Pyttsx3 (Text-to-Speech)

---

## How to Run the Project

1. Clone the repository
   ```bash
   git clone https://github.com/chetan-ai037/pdf-voice-chatbot.git
   cd pdf-voice-chatbot
2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
3. Create a .env file and add your API key
   ```bash
  GROQ_API_KEY=your_api_key_here
4. Run the application
   ```bash
   streamlit run pdf_voice_chatbot/app.py

