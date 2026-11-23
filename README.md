# Company Policy Assistant

A chatbot that answers employee questions about company policies, built with
LangChain and OpenAI. Policy documents live in `data/`.

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
export OPENAI_API_KEY=...   # see .env.example
```

## Run

```bash
.venv/bin/python agent.py           # chat
.venv/bin/python agent.py --debug   # chat, showing tool calls
```
