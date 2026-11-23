"""Search the policy documents directly, without an LLM.

Run (after ingest.py; needs OPENAI_API_KEY):
    .venv/bin/python search.py "How many vacation days do I get?"
"""

import sys

from vector_store import get_vector_store

TOP_K = 3


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        sys.exit('Usage: .venv/bin/python search.py "your question"')

    # Embeds the question, then finds the chunks with the closest vectors.
    results = get_vector_store().similarity_search_with_score(question, k=TOP_K)
    if not results:
        sys.exit("No chunks found. Run ingest.py first.")

    for rank, (chunk, distance) in enumerate(results, start=1):
        page = chunk.metadata.get("page", "-")
        print(f"{rank}. {chunk.metadata['source']} page {page}  distance {distance:.3f} (lower is closer)")
        print(f"{chunk.page_content}\n")


if __name__ == "__main__":
    main()
