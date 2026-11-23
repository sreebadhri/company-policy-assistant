"""A simple LangChain agent backed by OpenAI, with two HR-policy tools.

Run (needs OPENAI_API_KEY; override the model with OPENAI_MODEL):
    .venv/bin/python agent.py            # chat
    .venv/bin/python agent.py --debug    # chat, showing tool calls
"""

import os
import sys
import uuid

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# Stand-in for a real policy store (later: swap for a vector-store retriever).
POLICIES = {
    "vacation": "Full-time employees accrue 15 vacation days per year, "
    "increasing to 20 days after 3 years of service and 25 after 7 years.",
    "sick leave": "Employees receive 10 paid sick days per year. "
    "A doctor's note is required for absences longer than 3 consecutive days.",
    "remote work": "Employees may work remotely up to 3 days per week "
    "with manager approval.",
    "parental leave": "Primary caregivers receive 16 weeks of paid leave; "
    "secondary caregivers receive 6 weeks.",
}


@tool
def lookup_policy(topic: str) -> str:
    """Look up the company HR policy for a topic.

    Known topics: vacation, sick leave, remote work, parental leave.
    """
    key = topic.lower().strip()
    for name, text in POLICIES.items():
        if name in key or key in name:
            return text
    return f"No policy found for '{topic}'. Known topics: {', '.join(POLICIES)}."


@tool
def calculate_days(total_days: float, days_used: float) -> float:
    """Return how many leave days remain given the allowance and days already used."""
    return total_days - days_used


model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))

agent = create_agent(
    model=model,
    tools=[lookup_policy, calculate_days],
    system_prompt=(
        "You are an HR assistant. Answer questions using the lookup_policy tool; "
        "do not invent policy details. Use calculate_days for leave arithmetic. "
        "Keep answers short."
    ),
    # Keeps the conversation history so follow-up questions have context.
    checkpointer=InMemorySaver(),
)


def ask(question: str, thread_id: str, show_tools: bool) -> str:
    """Send one user message in the given conversation and return the reply."""
    config = {"configurable": {"thread_id": thread_id}}
    before = len(agent.get_state(config).values.get("messages", []))
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config
    )
    if show_tools:
        for msg in result["messages"][before:]:
            if msg.type == "ai" and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"  [tool] {call['name']}({call['args']})")
            elif msg.type == "tool":
                print(f"  [result] {msg.content}")
    return result["messages"][-1].text


def main() -> None:
    show_tools = "--debug" in sys.argv
    thread_id = str(uuid.uuid4())
    print("HR Assistant - ask me about company policies.")
    print("Commands: /new (start over), /quit (exit)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break
        if question.lower() == "/new":
            thread_id = str(uuid.uuid4())
            print("(Started a new conversation)\n")
            continue
        try:
            print(f"Bot: {ask(question, thread_id, show_tools)}\n")
        except Exception as e:  # keep the chat alive on API/network errors
            print(f"Bot: Sorry, something went wrong: {e}\n")


if __name__ == "__main__":
    main()
