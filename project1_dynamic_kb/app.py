import streamlit as st
from rag_chain import get_rag_response
# from scheduler import start_scheduler

# Start the background scheduler once, when the app first loads
# if "scheduler_started" not in st.session_state:
#     start_scheduler()
#     st.session_state.scheduler_started = True

import streamlit as st
from rag_chain import get_rag_response
from scheduler import start_scheduler
from ingestion import get_all_chunks
from vector_store import add_chunks_to_store

# One-time ingestion on cold start (needed for cloud deployment where chroma_db isn't persisted)
if "kb_initialized" not in st.session_state:
    with st.spinner("Setting up knowledge base..."):
        chunks = get_all_chunks()
        add_chunks_to_store(chunks)
    st.session_state.kb_initialized = True

# Start the background scheduler once, when the app first loads
if "scheduler_started" not in st.session_state:
    start_scheduler()
    st.session_state.scheduler_started = True

st.set_page_config(page_title="Dynamic Knowledge Base Chatbot", page_icon="🤖")
st.title("🤖 Dynamic Knowledge Base Chatbot")
st.caption("Ask questions — answers are grounded in documents from your knowledge base, which updates automatically.")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box
user_question = st.chat_input("Ask a question...")

if user_question:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = get_rag_response(user_question)
            answer = result["answer"]
            sources = result["sources"]
            sentiment = result["sentiment"]
            language = result["language"]

            sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}[sentiment]
            st.caption(f"{sentiment_emoji} Sentiment: {sentiment} | 🌐 Language: {language}")
            
            

            st.write(answer)
            if sources:
                st.caption(f"📄 Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})