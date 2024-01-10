import requests
from clarifai.client.model import Model
from langchain_community.llms import Clarifai
from langchain.agents import initialize_agent, load_tools, Tool, create_structured_chat_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

USER_ID = 'openai'
APP_ID = 'chat-completion'
PAT = '9c5de2311a9f4ee998f1a28a9bfea9d5'
MODEL_ID = 'gpt-4-turbo'
MODEL_VERSION_ID = '182136408b4b4002a920fd500839f2c8'
RAW_TEXT = 'Who won the worldcup cricket match in 2023?'

clarifai_llm = Clarifai(pat=PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)

prompt = PromptTemplate(
    input_variables=["query"],
    template = "You are New Native Internal Bot. Help users with their important tasks, like a professor in a particular field. Query: {query}"
)

user_input = input("Enter the question: ")

# llm_chain = LLMChain(llm = clarifai_llm, prompt=prompt)
# print(llm_chain.invoke(f"{user_input}")['text'])

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

r_1 = agent(f"{user_input}")
print(f"Final answer: {r_1['output']}")