import os
import streamlit as st

from vector import build_index
from main import answer_question

st.title("PDF Q&A Bot")

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if uploaded_file.name != st.session_state.last_uploaded_file:

        pdf_name = os.path.splitext(
            uploaded_file.name
        )[0]

        pdf_path = f"temp_{uploaded_file.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        num_chunks = build_index(
            pdf_path,
            pdf_name
        )

        st.session_state.last_uploaded_file = uploaded_file.name

        st.success(
            f"Indexed {num_chunks} chunks from {uploaded_file.name}"
        )

prompt = st.chat_input(
    "Ask a question"
)

if prompt:

    st.chat_message("user").write(prompt)

    answer = answer_question(prompt)

    st.chat_message("assistant").write(answer)