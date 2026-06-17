import streamlit as st
from main import answer_question
st.title("Echo bot")


with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )


prompt = st.chat_input("Ask a question about the PDF")

if prompt:
    st.chat_message("user").write(prompt)

    answer = answer_question(prompt) 
    st.chat_message("assistant").write(answer)

if uploaded_file:

    from pypdf import PdfReader

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.write("Pages:", len(pdf.pages))
    st.text(text[:3000])