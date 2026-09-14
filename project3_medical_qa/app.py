import streamlit as st
from rag_chain import generate_answer

st.set_page_config(page_title="Medical Q&A Chatbot", page_icon="🩺")
st.title("🩺 Medical Q&A Chatbot")
st.caption("Powered by MedQuAD dataset. Not a substitute for professional medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(query)
            st.markdown(answer)
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"**{s['metadata']['focus']}** ({s['metadata']['qtype']}) — distance: {s['distance']:.3f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})