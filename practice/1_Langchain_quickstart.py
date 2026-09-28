# run ==> streamlit run 1_Langchain_Quickstart.py

import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Langchain Quickstart App")
 
api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("MODEL")

def generate_response(input_text):
    chat = ChatOpenAI(
        model=llm_model,
        temperature=0.7,
        max_tokens=500,
        base_url="https://openrouter.ai/api/v1"
    )
    resp = chat.invoke(input_text)
    st.write(resp.content)
    
with st.form("my_form"):
    text = st.text_area(
        "Enter text:", ""
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)