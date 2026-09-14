import streamlit as st
import ollama
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Multi-modal AI Assistant", page_icon="🖼️")
st.title("🖼️ Multi-modal AI Assistant")
st.caption("Upload an image and ask questions about it, or just chat normally.")

VISION_MODEL = "moondream"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image_b64" not in st.session_state:
    st.session_state.current_image_b64 = None

uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    st.session_state.current_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something about the image, or anything else...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.current_image_b64:
                response = ollama.chat(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": user_input,
                        "images": [st.session_state.current_image_b64]
                    }]
                )
            else:
                response = ollama.chat(
                    model=VISION_MODEL,
                    messages=st.session_state.messages
                )

            answer = response["message"]["content"]
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})