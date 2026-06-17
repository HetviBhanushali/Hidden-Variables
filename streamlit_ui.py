import streamlit as st

st.title("Echo bot")


with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )


prompt = st.chat_input("Ask a question about the PDF")

if prompt:
    st.chat_message("user").write(prompt)