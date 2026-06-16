import streamlit as st
import tempfile 
st.title("Echo Bot")
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User: {prompt}")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
if pdf_file:
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf") as temp:
        temp.write(pdf_file.getvalue())