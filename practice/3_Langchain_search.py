import streamlit as st
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
selected_model = os.getenv("MODEL")

with st.sidebar:
    serper_api_key = st.text_input(
        "Serper API Key", key="langchain_search_api_key_serper"
    )

st.title("🔎 Search with LangChain")
question = st.text_input(
    "What do you want to know?", placeholder="Who won the Women's U.S. Open in 2018?"
)

if question:
    if not serper_api_key and not api_key:
        st.info(
            "Please add your Serper API keys and connect with OpenRouter to continue."
        )
    elif not serper_api_key:
        st.info("Please add your Serper API key to continue.")
    elif not api_key:
        st.info("Please connect with OpenRouter to continue.")
    elif serper_api_key and api_key:
        llm = ChatOpenAI(
            model=selected_model,
            temperature=0.7,
            max_tokens=500,
            base_url="https://openrouter.ai/api/v1",
            # headers={"HTTP-Referer": "https://openrouter.ai/api/v1"},
        )

        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

        search_tool = Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search",
        )
        search_agent = initialize_agent(
            [search_tool],
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        response = search_agent.run(question)
        st.write("### Answer")
        st.write(response)