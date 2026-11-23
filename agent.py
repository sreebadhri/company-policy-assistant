"""Company policy assistant: a terminal chatbot built with LangChain and OpenAI.

The agent will answer from the company documents once document search is
added (Step 5 of the RAG plan). Until then it has no tools and says so.

Run (needs OPENAI_API_KEY; override the model with OPENAI_MODEL):
    .venv/bin/python agent.py            # chat
    .venv/bin/python agent.py --debug    # chat, showing tool calls
"""

import os
import sys
import uuid

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"

SYSTEM_PROMPT = (
    "You are the company policy assistant. Answer employee questions about "
    "company policies. You do not have access to the policy documents yet, so "
    "do not guess policy details: say you can't look them up yet. "
    "Keep answers short."
)

EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}


def build_agent():
    """Create the agent. The checkpointer keeps each conversation's history,
    so follow-up questions have context."""
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    return create_agent(
        model=model,
        tools=[],  # Step 5 adds the document search tool here.
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def print_tool_activity(messages) -> None:
    """Print the tool calls and results from one turn (used with --debug)."""
    for msg in messages:
        if msg.type == "ai" and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  [tool] {call['name']}({call['args']})")
        elif msg.type == "tool":
            print(f"  [result] {msg.content}")


def ask(agent, question: str, thread_id: str, debug: bool = False) -> str:
    """Send one user message in the given conversation and return the reply."""
    config = {"configurable": {"thread_id": thread_id}}
    history_length = len(agent.get_state(config).values.get("messages", []))

    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)

    if debug:
        print_tool_activity(result["messages"][history_length:])
    return result["messages"][-1].text


def main() -> None:
    debug = "--debug" in sys.argv
    agent = build_agent()
    thread_id = str(uuid.uuid4())

    print("Company Policy Assistant - ask me about company policies.")
    print("Commands: /new (start over), /quit (exit)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Bye!")
            break
        if question.lower() == "/new":
            thread_id = str(uuid.uuid4())
            print("(Started a new conversation)\n")
            continue

        try:
            print(f"Bot: {ask(agent, question, thread_id, debug)}\n")
        except Exception as error:  # keep the chat alive on API/network errors
            print(f"Bot: Sorry, something went wrong: {error}\n")


if __name__ == "__main__":
    main()
