import streamlit as st
st.title("Echo Bot")
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User: {prompt}")

