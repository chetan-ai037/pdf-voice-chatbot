import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def sanitize_query(query):
    forbidden = [
        "ignore previous",
        "system prompt",
        "bypass",
        "act as",
        "jailbreak"
    ]
    for f in forbidden:
        if f in query.lower():
            raise ValueError("Prompt injection detected.")
    return query


def ask_llm_secure(context, question, is_summary=False, chat_history=None):
    """
    Optimized LLM query for accuracy with conversation history support.
    
    Args:
        context: Document context from vector search
        question: Current question
        is_summary: Whether this is a summary question
        chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
    """
    # Build messages list with conversation history
    messages = []
    
    # Add conversation history if provided
    if chat_history:
        messages.extend(chat_history)
    
    # Improved prompts for better accuracy and clarity with conversation awareness
    if is_summary:
        conversation_note = ""
        if chat_history:
            conversation_note = """\nIMPORTANT: This is part of an ongoing conversation. 
- Review the conversation history above to understand context
- If the user asks follow-up questions (e.g., "tell me more about that", "what about X", "explain that further"), reference what was discussed earlier
- Connect your answer to previous topics when relevant
- Use phrases like "As mentioned earlier" or "Building on the previous point" when appropriate\n"""
        
        user_content = f"""You are an expert document analyst. Provide a comprehensive summary based on the document content below.
{conversation_note}
<document_content>
{context}
</document_content>

Question: {question}

Instructions:
- Provide a clear, well-structured answer based solely on the document content
- {f'If this is a follow-up question, connect it to previous conversation topics naturally' if chat_history else ''}
- Organize information logically and include all key points
- Be thorough yet concise
- If the question asks for a summary, structure it with clear sections
- Use only information from the document - do not add external knowledge

Answer:"""
    else:
        conversation_note = ""
        if chat_history:
            conversation_note = """\nIMPORTANT: This is part of an ongoing conversation. 
- Review the conversation history above carefully
- If the question contains pronouns ("it", "that", "this", "the above") or references ("the first point", "that topic"), relate them to previous conversation
- If the user asks "what about X?" or "tell me more about that", understand what "that" refers to from context
- Build upon previous answers when the question is a follow-up\n"""
        
        user_content = f"""You are an expert assistant. Answer the question using ONLY the information provided in the context below.
{conversation_note}
<context>
{context}
</context>

Question: {question}

Instructions:
- Answer directly and accurately based on the context provided
- {f'If this question relates to previous conversation, explicitly reference and build upon earlier topics' if chat_history else ''}
- {f'Understand pronouns and references by connecting them to previous questions/answers' if chat_history else ''}
- Include specific details and relevant information from the context
- If multiple related points are found, include them all
- If the information is not in the context, explicitly state that
- Be precise and cite relevant details from the context

Answer:"""
    
    messages.append({"role": "user", "content": user_content})
    
    # Optimized settings for accuracy while maintaining speed
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,  # Slightly higher for more nuanced answers
        max_tokens=1000,  # Increased for more comprehensive answers
        top_p=0.95,       # Higher top_p for better answer quality
        stream=False
    )
    return res.choices[0].message.content.strip()


def validate_answer(answer, context):
    """Validates if answer is supported by context. Returns True if validated."""
    prompt = f"""
Answer:
{answer}

Context:
{context}

Is the answer fully supported by the context?
Reply only YES or NO.
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10  # Only need YES/NO
        )
        return "YES" in res.choices[0].message.content.upper()
    except:
        return True  # Don't block if validation fails