# run ==> streamlit run 2_Langchain_PromptTemplate.py

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("MODEL")

def blog_outline(topic):
    # Instantiate LLM model
    llm_chat = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=llm_model,
        temperature=0.7,
        max_tokens=500
    )
    
    #----- วิธี 1 ---------
    # Prompt
    # template_system = "As an experienced data scientist and technical writer, generate an outline for a blog about {topic}."
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template_system)
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    # prompt = chat_prompt.format_prompt(topic=topic).to_messages()
    # st.write(llm_chat(prompt))
    
    #----- วิธี / ---------
    system = SystemMessage("As an experienced data scientist and technical writer, generate an outline for a blog about {topic}.")
    human = HumanMessage(topic)
    prompt = ChatPromptTemplate.from_messages([
        system,
        human
    ])
    chain = prompt | llm_chat
    response = chain.invoke({"topic":topic})
    st.write(response.content)
    
with st.form("myform"):
    topic_text = st.text_input("Enter prompt:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        blog_outline(topic_text)

    

