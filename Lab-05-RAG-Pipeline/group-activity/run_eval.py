"""
Runs every question in eval_questions.csv through the Lab 05 RAG (baseline)
and the Graph RAG, and saves the answers and response times.
"""
import csv
import time

from graph_rag import graph_ask, BASE_DIR      # also makes the Lab 05 files importable
from rag_pipeline import ask

QUESTIONS = BASE_DIR / "eval_questions.csv"
RESULTS = BASE_DIR / "results" / "eval_answers.csv"


def timed(function, question):
    start = time.perf_counter()
    result = function(question)
    return result["answer"], round(time.perf_counter() - start, 1)


if __name__ == "__main__":
    RESULTS.parent.mkdir(exist_ok=True)
    with open(QUESTIONS, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        baseline, baseline_time = timed(ask, q["question"])
        graph, graph_time = timed(graph_ask, q["question"])
        rows.append({**q, "baseline_answer": baseline, "baseline_seconds": baseline_time,
                     "graph_answer": graph, "graph_seconds": graph_time})

    with open(RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} results to {RESULTS}")