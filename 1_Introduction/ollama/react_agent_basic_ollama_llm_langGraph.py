from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOllama(model="llama3.1:8b", temperature=0)

search_tool = TavilySearchResults(search_depth="basic", max_results=5)

tools = [search_tool]
agent = create_react_agent(llm, tools)

result = agent.invoke(
    {"messages": [("user", "Give me the current IPL table")]},
    config={"recursion_limit": 10},
)
print(result["messages"][-1].content)