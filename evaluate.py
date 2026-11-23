"""Score search and answers against eval/questions.json.

Each answerable question lists the page that answers it and the facts a correct
answer must contain ("five|5" means either will do). Questions marked
"answerable": false are not covered by the documents: the bot should say so.

Run (after ingest.py; needs OPENAI_API_KEY):
    .venv/bin/python evaluate.py             # search only (cheap)
    .venv/bin/python evaluate.py --answers   # also ask the agent
"""

import json
import sys
import uuid
from pathlib import Path

from agent import TOP_K, ask, build_agent
from vector_store import get_vector_store

QUESTIONS_FILE = Path(__file__).parent / "eval" / "questions.json"
NOT_FOUND_PHRASES = ["couldn't find", "could not find", "can't find", "cannot find", "didn't find", "did not find"]


def is_answerable(case: dict) -> bool:
    return case.get("answerable", True)


def find_rank(case: dict, chunks) -> int | None:
    """Position (from 1) of the first chunk from the expected page, if any."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.metadata["source"] == case["source"] and chunk.metadata.get("page") == case["page"]:
            return rank
    return None


def answer_problems(case: dict, answer: str) -> list[str]:
    """Everything wrong with an answer. An empty list means it passed."""
    text = answer.lower().replace("’", "'")
    if not is_answerable(case):
        return [] if any(p in text for p in NOT_FOUND_PHRASES) else ["did not say it couldn't find it"]

    problems = [
        f"missing fact: {fact}"
        for fact in case["facts"]
        if not any(option.lower() in text for option in fact.split("|"))
    ]
    citation = f"{case['source']}, page {case['page']}"
    if citation not in text:
        problems.append(f"missing citation: {citation}")
    return problems


def evaluate_search(cases: list[dict]) -> bool:
    print(f"Search (expected page in the top {TOP_K}, using the question as written)")
    store = get_vector_store()
    ranks = []
    for case in filter(is_answerable, cases):
        rank = find_rank(case, store.similarity_search(case["question"], k=TOP_K))
        ranks.append(rank)
        print(f"  {'rank ' + str(rank) if rank else 'MISS  '}  {case['question']}")
    found = sum(rank is not None for rank in ranks)
    print(f"  => top {TOP_K}: {found}/{len(ranks)}, first: {ranks.count(1)}/{len(ranks)}\n")
    return found == len(ranks)


def evaluate_answers(cases: list[dict]) -> bool:
    print("Answers (facts and citation, or 'couldn't find it')")
    agent = build_agent()
    passed = {True: 0, False: 0}
    for case in cases:
        # A new thread per question, so earlier ones can't influence it.
        answer = ask(agent, case["question"], str(uuid.uuid4()))
        problems = answer_problems(case, answer)
        passed[is_answerable(case)] += not problems
        print(f"  {'FAIL' if problems else 'pass'}  {case['question']}")
        if problems:
            print(f"        {'; '.join(problems)}\n        answer: {answer}")
    answerable = sum(map(is_answerable, cases))
    print(f"  => answerable: {passed[True]}/{answerable}, not in documents: {passed[False]}/{len(cases) - answerable}")
    return sum(passed.values()) == len(cases)


def main() -> None:
    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    ok = evaluate_search(cases)
    if "--answers" in sys.argv:
        ok = evaluate_answers(cases) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
