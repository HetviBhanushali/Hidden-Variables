import streamlit as st
import tempfile

from vector import build_index
from main import answer_question

st.title("PDF Q&A Bot")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    build_index(pdf_path)

    st.success("PDF indexed successfully!")

prompt = st.chat_input(
    "Ask a question"
)

if prompt:

    st.chat_message("user").write(prompt)

    answer = answer_question(prompt)

    st.chat_message("assistant").write(answer)

if uploaded_file and "indexed" not in st.session_state:

    build_index(pdf_path)

    st.session_state.indexed = True