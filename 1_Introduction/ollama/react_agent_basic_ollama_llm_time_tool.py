from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOllama(model="llama3.1:8b", temperature=0)

search_tool = TavilySearchResults(search_depth="basic", max_results=5)


@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Return the current date and time for an IANA timezone.

    Examples of valid timezone values: 'UTC', 'Asia/Kolkata' (India),
    'America/Los_Angeles', 'Europe/London'. Use this tool whenever the
    user asks for the current time, date, or "now" in any location.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception as e:
        return f"Invalid timezone '{timezone}': {e}"
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z (%z)")


tools = [search_tool, get_current_datetime]
agent = create_react_agent(llm, tools)

result = agent.invoke(
    {"messages": [("user", "What is current date and time in India?")]},
    config={"recursion_limit": 10},
)
print(result["messages"][-1].content)