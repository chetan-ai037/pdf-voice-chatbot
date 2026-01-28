import streamlit as st
import datetime
import os
from utils.pdf_utils import extract_text_from_pdfs
from utils.vector_utils import build_vector_store, retrieve_secure, rerank_chunks
from utils.llm_utils import sanitize_query, ask_llm_secure, validate_answer
from utils.stt_utils import transcribe_audio
from utils.tts_utils import speak, save_to_file

# Create temp directory for audio files
if not os.path.exists("temp_audio"):
    os.makedirs("temp_audio")

st.set_page_config(
    page_title="Document Query System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Document Query System")

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat history management - store multiple chats
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}  # Dictionary: {chat_id: {title: str, messages: list, timestamp: str}}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "chat_0"
    st.session_state.all_chats["chat_0"] = {
        "title": "New Chat",
        "messages": [],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# Chat rename/delete UI state
if "rename_chat_id" not in st.session_state:
    st.session_state.rename_chat_id = None
if "rename_chat_value" not in st.session_state:
    st.session_state.rename_chat_value = ""

# Performance mode - disable expensive reranking and validation for speed
PERFORMANCE_MODE = True

# Ensure chat_history is synced with current chat on page load
# This ensures that when switching chats, the messages are displayed correctly
if "chat_switched" not in st.session_state:
    st.session_state.chat_switched = False

# Sync chat_history with current chat when switching (but not on every rerun)
if st.session_state.current_chat_id in st.session_state.all_chats:
    current_chat_data = st.session_state.all_chats[st.session_state.current_chat_id]
    stored_messages = current_chat_data.get("messages", [])
    # Only sync if stored messages exist and chat_history is empty or different
    # This prevents overwriting when user is actively chatting
    if stored_messages and (not st.session_state.chat_history or st.session_state.chat_switched):
        if len(stored_messages) != len(st.session_state.chat_history):
            st.session_state.chat_history = stored_messages.copy()
            st.session_state.chat_switched = False

# Main chat interface - unified layout
# Document upload section at the top
col1, col2 = st.columns([3, 1])
with col1:
    files = st.file_uploader(
        "**Upload PDF Documents**",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF documents for analysis. Drag and drop or click to browse. Multiple files supported.",
        label_visibility="visible"
    )
with col2:
    if st.session_state.index:
        if st.button("New Chat", use_container_width=True, type="primary"):
            # Save current chat to history if it has messages
            current_chat = st.session_state.all_chats.get(st.session_state.current_chat_id, {})
            if current_chat.get("messages") and len(current_chat["messages"]) > 0:
                # Generate a title from first user message
                first_user_msg = next((msg["content"] for msg in current_chat["messages"] if msg["role"] == "user"), None)
                if first_user_msg:
                    title = first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
                    current_chat["title"] = title
            
            # Create new chat
            new_chat_id = f"chat_{len(st.session_state.all_chats)}"
            st.session_state.current_chat_id = new_chat_id
            st.session_state.all_chats[new_chat_id] = {
                "title": "New Chat",
                "messages": [],
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.chat_history = []
            st.rerun()

# Sidebar for chat history
with st.sidebar:
    st.markdown("### 📚 Chat History")
    st.markdown("---")
    
    if st.session_state.index:
        # Rename mode UI (top of sidebar)
        if st.session_state.rename_chat_id is not None:
            chat_id = st.session_state.rename_chat_id
            existing = st.session_state.all_chats.get(chat_id, {})
            st.markdown("**Rename chat**")
            st.caption(f"Current: {existing.get('title', 'New Chat')}")
            st.session_state.rename_chat_value = st.text_input(
                "New name",
                value=st.session_state.rename_chat_value or existing.get("title", ""),
                key="rename_chat_input",
                label_visibility="collapsed",
                placeholder="Enter chat name..."
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save", use_container_width=True, type="primary", key="rename_chat_save"):
                    new_title = (st.session_state.rename_chat_value or "").strip()
                    if new_title:
                        st.session_state.all_chats[chat_id]["title"] = new_title[:80]
                    st.session_state.rename_chat_id = None
                    st.session_state.rename_chat_value = ""
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True, type="secondary", key="rename_chat_cancel"):
                    st.session_state.rename_chat_id = None
                    st.session_state.rename_chat_value = ""
                    st.rerun()
            st.markdown("---")

        # Show current chat info
        current_chat = st.session_state.all_chats.get(st.session_state.current_chat_id, {})
        if current_chat.get("messages"):
            st.markdown(f"**Current Chat:**")
            st.caption(f"💬 {len(current_chat.get('messages', []))} messages")
            st.markdown("---")
        
        # Show all chats
        if len(st.session_state.all_chats) > 0:
            # Sort chats by timestamp (newest first)
            sorted_chats = sorted(st.session_state.all_chats.items(), 
                                 key=lambda x: x[1]["timestamp"], 
                                 reverse=True)
            
            for chat_id, chat_data in sorted_chats:
                # Highlight current chat
                is_current = chat_id == st.session_state.current_chat_id
                button_style = "primary" if is_current else "secondary"
                
                if len(chat_data.get("messages", [])) > 0:
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col1:
                        chat_title = chat_data.get("title", "New Chat")
                        if is_current:
                            chat_title = f"▶ {chat_title}"
                        if st.button(chat_title, key=f"load_{chat_id}", use_container_width=True, type=button_style):
                            # Save current chat before switching
                            if st.session_state.current_chat_id in st.session_state.all_chats:
                                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.chat_history.copy()
                            
                            # Switch to selected chat
                            st.session_state.current_chat_id = chat_id
                            # Load messages from selected chat
                            selected_chat = st.session_state.all_chats.get(chat_id, {})
                            st.session_state.chat_history = selected_chat.get("messages", []).copy()
                            st.session_state.chat_switched = True
                            st.rerun()
                    with col2:
                        if st.button("✏️", key=f"rename_{chat_id}", help="Rename this chat"):
                            st.session_state.rename_chat_id = chat_id
                            st.session_state.rename_chat_value = chat_data.get("title", "")
                            st.rerun()
                    with col3:
                        if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                            del st.session_state.all_chats[chat_id]
                            if st.session_state.current_chat_id == chat_id:
                                # Switch to first available chat
                                if st.session_state.all_chats:
                                    st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0]
                                    st.session_state.chat_history = st.session_state.all_chats[st.session_state.current_chat_id]["messages"].copy()
                                else:
                                    st.session_state.current_chat_id = "chat_0"
                                    st.session_state.all_chats["chat_0"] = {
                                        "title": "New Chat",
                                        "messages": [],
                                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                                    }
                                    st.session_state.chat_history = []
                            st.rerun()
                    st.caption(f"📅 {chat_data.get('timestamp', 'N/A')} | 💬 {len(chat_data.get('messages', []))} messages")
                else:
                    # Show empty/new chats
                    if is_current:
                        st.markdown(f"**▶ {chat_data.get('title', 'New Chat')}** (Current)")
                    else:
                        col1, col2, col3 = st.columns([6, 1, 1])
                        with col1:
                            load_clicked = st.button(chat_data.get('title', 'New Chat'), key=f"load_{chat_id}", use_container_width=True)
                        with col2:
                            if st.button("✏️", key=f"rename_empty_{chat_id}", help="Rename this chat"):
                                st.session_state.rename_chat_id = chat_id
                                st.session_state.rename_chat_value = chat_data.get("title", "")
                                st.rerun()
                        with col3:
                            if st.button("🗑️", key=f"delete_empty_{chat_id}", help="Delete this chat"):
                                del st.session_state.all_chats[chat_id]
                                if st.session_state.current_chat_id == chat_id:
                                    if st.session_state.all_chats:
                                        st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0]
                                        st.session_state.chat_history = st.session_state.all_chats[st.session_state.current_chat_id].get("messages", []).copy()
                                    else:
                                        st.session_state.current_chat_id = "chat_0"
                                        st.session_state.all_chats["chat_0"] = {
                                            "title": "New Chat",
                                            "messages": [],
                                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                                        }
                                        st.session_state.chat_history = []
                                st.rerun()
                        if load_clicked:
                            # Save current chat before switching
                            if st.session_state.current_chat_id in st.session_state.all_chats:
                                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.chat_history.copy()
                            
                            # Switch to selected chat
                            st.session_state.current_chat_id = chat_id
                            # Load messages from selected chat (should be empty for new chats)
                            selected_chat = st.session_state.all_chats.get(chat_id, {})
                            st.session_state.chat_history = selected_chat.get("messages", []).copy()
                            st.session_state.chat_switched = True
                            st.rerun()
        else:
            st.info("No chat history yet. Start a conversation to see it here.")
    else:
        st.info("Upload documents to start chatting.")

# Process uploaded files
if files:
    # Only process if we don't already have an index or if files have changed
    # Check if files are new by comparing with stored file names
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    # Get current file names
    current_file_names = [f.name for f in files] if files else []
    
    # Only process if files are different or no index exists
    if not st.session_state.index or set(current_file_names) != set(st.session_state.processed_files):
        with st.spinner("Processing documents..."):
            try:
                text = extract_text_from_pdfs(files)
                if not text or len(text.strip()) == 0:
                    st.error("No text could be extracted from the PDF files. Please ensure the PDFs contain readable text.")
                else:
                    with st.spinner("Building search index..."):
                        index, chunks = build_vector_store(text)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.processed_files = current_file_names
                    # Reset to new chat when new PDF is uploaded
                    st.session_state.current_chat_id = "chat_0"
                    st.session_state.all_chats = {
                        "chat_0": {
                            "title": "New Chat",
                            "messages": [],
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                    }
                    st.session_state.chat_history = []
                    st.success(f"**Documents indexed successfully.** Indexed **{len(chunks)}** document segments. Ready for queries.")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.session_state.index = None
                st.session_state.chunks = None

# Display status if documents are indexed
if st.session_state.index:
    st.caption(f"📄 **{len(st.session_state.chunks)}** segments indexed | 💬 **{len(st.session_state.chat_history)}** messages")

# Empty state when no documents uploaded
if not st.session_state.index:
    st.info("👆 **Upload PDF documents above to begin querying.**")
    st.stop()

# Initialize transcribed question in session state
if "transcribed_question" not in st.session_state:
    st.session_state.transcribed_question = None

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# Voice input section - moved to bottom
audio = None
# Will be rendered after chat history

# Query input - main chat input
question = st.chat_input("Enter your query or use voice input above...")

# Check if we have a transcribed question from previous run
if st.session_state.transcribed_question:
    question = st.session_state.transcribed_question
    st.session_state.transcribed_question = None  # Clear it after using

# Cleaned up old audio processing block
# (Logic moved to bottom)

# Process new question first (before displaying history to avoid duplicates)
# Note: question can come from either text input or audio transcription
is_new_question = False
q = None

if question:
    q = sanitize_query(question)

    # Check if this is a duplicate question (avoid reprocessing on rerun)
    is_new_question = True
    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        
        # Check if we just answered this question (last message is assistant)
        if last_msg["role"] == "assistant":
            if len(st.session_state.chat_history) > 1:
                last_user_msg = st.session_state.chat_history[-2]
                if last_user_msg["role"] == "user" and last_user_msg["content"] == q:
                    # We already asked this and got an answer
                    is_new_question = False
        
        # Check if the last message is the same question (to avoid duplicates on rerun)
        elif last_msg["role"] == "user" and last_msg["content"] == q:
            is_new_question = False
    
    if is_new_question:
        # Add user question to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": q})
        # Update current chat in all_chats
        if st.session_state.current_chat_id in st.session_state.all_chats:
            st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.chat_history.copy()
            # Update title if this is the first message
            if len(st.session_state.chat_history) == 1:
                title = q[:50] + "..." if len(q) > 50 else q
                st.session_state.all_chats[st.session_state.current_chat_id]["title"] = title

# Display all query history (including any new question just added)
if st.session_state.chat_history:
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{msg['content']}**")
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                
                # Audio player implementation
                audio_key = f"audio_{i}"
                
                # Check if we have active audio for this message
                if audio_key in st.session_state:
                    st.audio(st.session_state[audio_key], format="audio/wav")
                    # Option to close/clear audio
                    if st.button("Close Audio", key=f"close_{i}", type="secondary"):
                        del st.session_state[audio_key]
                        st.rerun()
                else:
                    if st.button("Audio", key=f"speak_{i}", help="Listen to this response"):
                        with st.spinner("Generating audio..."):
                            # Create unique filename based on content hash or index
                            file_path = os.path.join("temp_audio", f"msg_{i}_{hash(msg['content'])}.wav")
                            
                            if not os.path.exists(file_path):
                                save_to_file(msg["content"], file_path)
                            
                            st.session_state[audio_key] = file_path
                            st.rerun()

# Render voice input at the bottom of chat history (so it's always visible near input)
if st.session_state.index:
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        audio = st.audio_input("Voice", label_visibility="collapsed", help="Click to record your voice query")
    with col2:
        st.caption("Click the microphone icon to record a voice query, or type your question below")
        
# Process audio input (if provided)
if question is None or question.strip() == "":
    if audio is not None:
        try:
            # Read audio bytes - handle errors gracefully
            try:
                audio_bytes = audio.read()
            except AttributeError:
                # Audio object doesn't have read method (widget error)
                audio_bytes = None
                # Silently ignore - widget error doesn't mean our processing failed
            except Exception as read_error:
                # Only show error if it's a real issue, not just widget display error
                error_str = str(read_error).lower()
                if "error" not in error_str or len(error_str) < 10:
                    # Likely just widget error, try to proceed silently
                    audio_bytes = None
                else:
                    # Real error - inform user
                    st.error(f"Failed to read audio: {str(read_error)}. Please check microphone permissions and try again.")
                    question = None
                    audio_bytes = None
            if audio_bytes is not None:
                # Check if we've already processed this exact audio file
                current_audio_hash = hash(audio_bytes)
                if st.session_state.last_audio_hash == current_audio_hash:
                    # Skip processing if we just handled this audio
                    question = None
                # Check if audio is not empty (minimum size check - very small files are likely empty)
                # Increased threshold for better quality audio detection: 3000 bytes (~0.1s of audio)
                elif len(audio_bytes) > 3000:
                    try:
                        # Store hash to prevent re-processing loop
                        st.session_state.last_audio_hash = current_audio_hash
                        
                        with st.spinner("Processing audio transcription..."):
                            transcribed_text, language = transcribe_audio(audio_bytes)
                        
                        if transcribed_text and transcribed_text.strip():
                            question = transcribed_text.strip()
                            # Store transcribed question in session state and rerun to reset audio widget
                            st.session_state.transcribed_question = question
                            # Show clear success message
                            st.success("**Voice transcription successful.**")
                            st.info(f"**Query:** {question}")
                            # Rerun to reset audio input widget
                            st.rerun()
                        else:
                            st.warning(" **No speech detected** in the recording. Please speak clearly for at least 2-3 seconds.")
                            question = None
                            # Rerun to reset audio widget even on failure
                            st.rerun()
                    except RuntimeError as e:
                        error_msg = str(e)
                        # Provide user-friendly error messages
                        if "Failed to load Whisper model" in error_msg:
                            st.error("**Speech recognition model failed to load.** Please wait a moment and try again.")
                        elif "Audio transcription failed" in error_msg:
                            # Extract the actual error from the RuntimeError
                            actual_error = error_msg.replace("Audio transcription failed: ", "")
                            if "Unsupported audio format" in actual_error:
                                st.error("**Audio format not supported.** Please try recording again or use text input.")
                            elif "Failed to convert audio" in actual_error:
                                st.error("**Could not process audio.** Please check microphone permissions and try recording again.")
                            elif "No speech detected" in actual_error:
                                st.warning("**No speech was detected.** Please speak clearly into your microphone for at least 2-3 seconds.")
                            else:
                                st.error(f"**Transcription failed:** {actual_error}. Please try again or use text input.")
                        else:
                            st.error(f"**Error:** {error_msg}. Please try again or use text input.")
                        question = None
                        # Rerun to reset audio widget on error
                        st.rerun()
                    except Exception as transcribe_error:
                        error_msg = str(transcribe_error)
                        st.error(f" Transcription error: {error_msg}. Please try again or use text input.")
                        question = None
                        # Rerun to reset audio widget on error
                        st.rerun()
                elif audio_bytes and len(audio_bytes) <= 3000:
                    # Very small audio - likely just silence or recording artifact
                    st.info("**Audio too short or too quiet.** Please speak clearly for at least 2-3 seconds.")
                    question = None
                    # Rerun to reset audio widget
                    st.rerun()
                else:
                    st.warning("**No audio data received.** Please check your microphone and try recording again.")
                    question = None
                    # Rerun to reset audio widget
                    st.rerun()
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = str(e)
            st.error(f"**Unexpected error:** {error_msg}. Please try again or use text input.")
            question = None
            # Rerun to reset audio widget on error
            st.rerun()

# Process new question to generate answer
if question and is_new_question:
    # Show spinner while processing and generate answer
    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            # Improved detection for summarization and overview questions
            summary_keywords = [
                "summar", "summary", "summarize", "summarise",
                "overview", "over all", "overall",
                "main", "key point", "key concept", "key idea",
                "explain", "what is this", "what is the",
                "chapter", "topics", "subject matter",
                "tell me about", "describe", "what does",
                "gist", "essence", "brief"
            ]
            
            is_doc_level = any(kw in q.lower() for kw in summary_keywords)

            # Handle document-level questions (summaries, overviews)
            # Improved sampling for better accuracy and comprehensive coverage
            if is_doc_level:
                total = len(st.session_state.chunks)

                # Enhanced sampling: more chunks for better accuracy
                if total <= 20:
                    # Small documents: use all chunks for complete coverage
                    selected = st.session_state.chunks
                elif total <= 50:
                    # Medium: use 16-18 chunks for comprehensive summary
                    step = total // 4
                    selected = (
                        st.session_state.chunks[:5] +
                        st.session_state.chunks[step:step+4] +
                        st.session_state.chunks[step*2:step*2+4] +
                        st.session_state.chunks[step*3:step*3+4] +
                        st.session_state.chunks[-5:]
                    )
                else:
                    # Large: use 18-22 chunks for thorough summary coverage
                    step = total // 5
                    selected = (
                        st.session_state.chunks[:5] +
                        st.session_state.chunks[step:step+4] +
                        st.session_state.chunks[step*2:step*2+4] +
                        st.session_state.chunks[step*3:step*3+4] +
                        st.session_state.chunks[step*4:step*4+4] +
                        st.session_state.chunks[-5:]
                    )

                context = "\n\n".join(selected)
                scores = [0.3]  # Higher confidence for doc-level questions

            else:
                # Specific/detailed questions: use semantic search
                # Increased k=10 for better accuracy and comprehensive context
                retrieved, scores = retrieve_secure(
                    q,
                    st.session_state.index,
                    st.session_state.chunks,
                    k=10  # More chunks for better accuracy and context coverage
                )

                if not retrieved or len(retrieved) == 0:
                    st.warning("**No relevant content found.** Please try rephrasing your query or asking about a different topic.")
                    st.stop()

                # Improved threshold handling for accuracy
                # Filter out very low relevance chunks but keep good ones
                filtered_retrieved = []
                filtered_scores = []
                for chunk, score in zip(retrieved, scores):
                    if score <= 0.9:  # Only filter out very unrelated chunks
                        filtered_retrieved.append(chunk)
                        filtered_scores.append(score)
                
                if not filtered_retrieved:
                    # If all chunks are filtered, use original but warn
                    filtered_retrieved = retrieved[:8]
                    filtered_scores = scores[:8]
                    st.warning("**Note:** The document may not contain specific information about this query. Generating response from available context...")
                else:
                    # Use top 7-8 most relevant chunks for comprehensive answers
                    filtered_retrieved = filtered_retrieved[:8]
                    filtered_scores = filtered_scores[:8]
                
                # Update scores for confidence calculation
                scores = filtered_scores
                context = "\n\n".join(filtered_retrieved)

            # Prepare chat history for LLM (last 6 exchanges to maintain context)
            # Exclude the current question (last item) since it's already in the prompt
            chat_history_for_llm = []
            if len(st.session_state.chat_history) > 1:
                # Get previous messages (excluding the current question we just added)
                # Get last 12 messages (6 Q&A pairs) but exclude the last user message
                previous_messages = st.session_state.chat_history[:-1]  # Exclude current question
                recent_messages = previous_messages[-12:] if len(previous_messages) > 12 else previous_messages
                for msg in recent_messages:
                    if msg["role"] in ["user", "assistant"]:
                        chat_history_for_llm.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
            # Generate answer with conversation history
            answer = ask_llm_secure(
                context, 
                q, 
                is_summary=is_doc_level,
                chat_history=chat_history_for_llm if chat_history_for_llm else None
            )

            # Calculate confidence based on best score (always calculate)
            if scores:
                confidence = round(1 / (1 + min(scores)), 2)
            else:
                confidence = 0.5

            # Skip validation in performance mode or for doc-level questions
            if not PERFORMANCE_MODE and not is_doc_level:
                try:
                    if not validate_answer(answer, context):
                        st.warning(" **Note:** Answer validation suggests some information may not be directly in the document.")
                except:
                    pass  # Don't block if validation fails

        # Add assistant response to chat history immediately
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "assistant":
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "confidence": confidence
            })
            # Update current chat in all_chats
            if st.session_state.current_chat_id in st.session_state.all_chats:
                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.chat_history.copy()

        # Log the interaction
        try:
            with open("audit.log", "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()} | {q} | {confidence}\n")
        except:
            pass  # Don't fail if logging fails
            
        # Rerun to display the new message via the main history loop
        st.rerun()
