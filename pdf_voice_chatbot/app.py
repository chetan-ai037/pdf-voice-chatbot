import streamlit as st
from utils.pdf_utils import extract_text_from_pdfs
from utils.vector_utils import build_vector_store, retrieve_secure, rerank_chunks
from utils.llm_utils import sanitize_query, rewrite_query, ask_llm_secure, validate_answer
from utils.stt_utils import transcribe_audio
from utils.tts_utils import speak
import datetime

st.set_page_config(page_title="Advanced Secure RAG", layout="centered")
st.title("Secure RAG  Chatbot")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    with st.spinner("Processing PDFs securely..."):
        text = extract_text_from_pdfs(files)
        index, chunks = build_vector_store(text)
        st.session_state.index = index
        st.session_state.chunks = chunks
    st.success("Documents indexed securely")

question = st.text_input("Ask a question (text)")
audio = st.audio_input("🎙️ Or speak your question")

if audio:
    question, lang = transcribe_audio(audio.read())
    st.info(f"You said: {question}")

if question and st.session_state.index:
    try:
        q = sanitize_query(question)
        q = rewrite_query(q)

        retrieved, scores = retrieve_secure(q, st.session_state.index, st.session_state.chunks)

        if min(scores) > 1.2:
            st.warning("Insufficient evidence in uploaded documents.")
            st.stop()

        ranked = rerank_chunks(q, retrieved)
        context = "\n\n".join(ranked)

        answer = ask_llm_secure(context, q)

        if not validate_answer(answer, context):
            st.warning("Answer could not be securely validated.")
            st.stop()

        confidence = round(1 / (1 + min(scores)), 2)

        with open("audit.log", "a") as f:
            f.write(f"{datetime.datetime.now()} | {q} | {confidence}\n")

        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Confidence Score: {confidence}")

        if st.button("🔊 Speak Answer"):
            speak(answer)

    except ValueError as e:
        st.error(str(e))
