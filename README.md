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
.venv/bin/python ingest.py          # build the vector database (run this first)
.venv/bin/python agent.py           # chat
.venv/bin/python agent.py --debug   # chat, showing tool calls
.venv/bin/python search.py "How many vacation days do I get?"   # search only, no LLM
```

Run `ingest.py` again whenever the documents in `data/` change. The document
text and your questions are sent to the OpenAI API.
