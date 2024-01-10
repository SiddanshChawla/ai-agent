import requests
from langchain_community.llms import Clarifai
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st
import os

# UI
st.title("Need an LLM with internet?")
user_input = st.text_input("Enter your query...")

clarifai_llm = Clarifai(pat=st.secrets["PAT"], user_id=st.secrets["USER_ID"], app_id=st.secrets["APP_ID"], model_id=st.secrets["MODEL_ID"])

prompt = PromptTemplate(
    input_variables=["query"],
    template = "You are New Native Internal Bot. Help users with their important tasks, like a professor in a particular field. Query: {query}"
)

webSearch = DuckDuckGoSearchRun()

search_tool = Tool(
    name = "Web Search",
    func = webSearch.run,
    description="A helpful tool to answer questions which require more upto date information or real time data."
)

agent = initialize_agent(
    agent="chat-zero-shot-react-description",
    tools=[search_tool],
    llm=clarifai_llm,
    verbose=True,
    max_iterations=3
)

if st.button('search'):
    with st.spinner("Loading"):
        r_1 = agent(f"{user_input}")
        st.write(f"Final answer: {r_1['output']}")

st.divider()