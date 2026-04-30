from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults

from langchain_classic.agents import initialize_agent, AgentType
load_dotenv()

llm = ChatOllama(model="llama3.1:8b", temperature=0.7)

search_tool = TavilySearchResults(search_depth="basic")


tools = [search_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

result= agent.invoke("Give me the current IPL table")
print(result)
# result = llm.invoke("what time is now?")

# print(result)