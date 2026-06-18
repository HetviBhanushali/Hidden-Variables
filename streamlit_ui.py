import os
import streamlit as st

from vector import build_index
from main import answer_question

st.title("PDF Q&A Bot")

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Always keep collection name available
    pdf_name = os.path.splitext(
        uploaded_file.name
    )[0]

    st.session_state.collection_name = pdf_name

    if uploaded_file.name != st.session_state.last_uploaded_file:

        pdf_path = f"temp_{uploaded_file.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        num_chunks = build_index(
            pdf_path,
            pdf_name
        )

        st.session_state.last_uploaded_file = uploaded_file.name

        st.success(
            "pdf loaded successfully"
        )

prompt = st.chat_input(
    "Ask a question"
)

if prompt:

    if st.session_state.collection_name is None:
        st.error("Please upload a PDF first.")
    else:
        st.chat_message("user").write(prompt)

        answer = answer_question(
            prompt,
            st.session_state.collection_name
        )

        st.chat_message("assistant").write(answer)