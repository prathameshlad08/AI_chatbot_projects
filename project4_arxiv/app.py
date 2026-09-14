import streamlit as st
from rag_chain import generate_answer

st.set_page_config(page_title="arXiv CS Research Assistant", page_icon="📚")
st.title("📚 arXiv CS Research Assistant")
st.caption("Ask about recent AI, ML, NLP, and CV research.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about CS research..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            answer, sources = generate_answer(query)
            st.markdown(answer)
            if sources:
                with st.expander(f"Papers referenced ({len(sources)})"):
                    for s in sources:
                        st.markdown(f"**{s['metadata']['title']}** — {s['metadata']['category']} ({s['metadata']['published'][:10]})")

    st.session_state.messages.append({"role": "assistant", "content": answer})