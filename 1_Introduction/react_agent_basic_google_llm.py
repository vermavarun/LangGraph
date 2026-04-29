from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
#from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import TavilySearchResults,GoogleSearchResults

from langchain_classic.agents import initialize_agent, AgentType
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

search_tool = TavilySearchResults(search_depth="basic")


tools = [search_tool]
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.invoke("Give me news for today covering india?")
# result = llm.invoke("what time is now?")

# print(result)