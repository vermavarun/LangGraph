from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # full conversation log
    iteration: int                                         # reflection loop counter

MAX_ITERATIONS = 3

# ── Nodes ─────────────────────────────────────────────────────────────────────

def generate(state: State) -> State:
    """Generate or improve the tweet."""
    system = (
        "You are a tweet writing assistant. "
        "Write or rewrite the tweet based on the conversation. "
        "Keep it under 280 characters, engaging, and on-topic. "
        "Return ONLY the tweet text, no extra explanation."
    )
    response = llm.invoke([HumanMessage(content=system)] + state["messages"])
    return {"messages": [AIMessage(content=response.content)], "iteration": state["iteration"]}


def reflect(state: State) -> State:
    """Critique the latest tweet and suggest improvements."""
    system = (
        "You are a social-media editor. "
        "Review the tweet and provide concise, actionable critique: "
        "clarity, tone, hashtags, engagement hooks, character count. "
        "Do NOT rewrite the tweet — only give feedback."
    )
    response = llm.invoke([HumanMessage(content=system)] + state["messages"])
    return {
        "messages": [HumanMessage(content=response.content)],
        "iteration": state["iteration"] + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def should_continue(state: State) -> str:
    """Loop back to generate until we hit MAX_ITERATIONS."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "reflect"


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(State)

builder.add_node("generate", generate)
builder.add_node("reflect", reflect)

builder.set_entry_point("generate")

builder.add_conditional_edges(
    "generate",
    should_continue,
    {"reflect": "reflect", "end": END},
)
builder.add_edge("reflect", "generate")

graph = builder.compile()

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Mermaid ──────────────────────────────────────────────")
    print(graph.get_graph().draw_mermaid())
    print("── ASCII ────────────────────────────────────────────────")
    graph.get_graph().print_ascii()
    print("─────────────────────────────────────────────────────────\n")

    topic = "Excited to share that I just launched my first open-source project on GitHub!"

    initial_state: State = {
        "messages": [HumanMessage(content=f"Write a tweet about: {topic}")],
        "iteration": 0,
    }

    print(f"Topic: {topic}\n{'='*60}")

    for step in graph.stream(initial_state):
        node, output = next(iter(step.items()))
        last_msg = output["messages"][-1]
        iteration = output["iteration"]

        if node == "generate":
            print(f"\n[Iteration {iteration}] GENERATED TWEET:\n{last_msg.content}")
        elif node == "reflect":
            print(f"\n[Iteration {iteration}] REFLECTION / CRITIQUE:\n{last_msg.content}")

    print("\n" + "="*60)
    print("Final tweet delivered after reflection loop.")
