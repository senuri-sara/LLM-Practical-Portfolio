"""
Human evaluation: shows each answer next to the reference answer and asks for a score.
  y = correct (1), p = partly correct (0.5), n = wrong (0)
The two answers are shown in random order without their method name, so the scoring is blind.
"""
import csv
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANSWERS = BASE_DIR / "results" / "eval_answers.csv"
SCORED = BASE_DIR / "results" / "eval_scored.csv"
SUMMARY = BASE_DIR / "results" / "summary.txt"
POINTS = {"y": 1.0, "p": 0.5, "n": 0.0}


def ask_score():
    while True:
        answer = input("  Score (y = correct, p = partly, n = wrong): ").strip().lower()
        if answer in POINTS:
            return POINTS[answer]


def average(values):
    return round(sum(values) / len(values), 2) if values else 0


if __name__ == "__main__":
    random.seed(42)
    with open(ANSWERS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, start=1):
        print("\n" + "=" * 80)
        print(f"Q{i}: {row['question']}")
        print(f"Reference: {row['reference_answer']}")
        methods = ["baseline", "graph"]
        random.shuffle(methods)
        for label, method in zip(["A", "B"], methods):
            print(f"\nAnswer {label}: {row[method + '_answer']}")
            row[method + "_score"] = ask_score()

    with open(SCORED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = ["Method        | Single-hop | Multi-hop | Overall | Avg time (s)",
             "------------- | ---------- | --------- | ------- | ------------"]
    for method, label in [("baseline", "Lab 05 RAG"), ("graph", "Graph RAG")]:
        scores = {hop: [float(r[method + "_score"]) for r in rows if r["hops"].strip().lower() == hop]
                  for hop in ["single", "multi"]}
        overall = [float(r[method + "_score"]) for r in rows]
        seconds = [float(r[method + "_seconds"]) for r in rows]
        lines.append(f"{label:<13} | {average(scores['single']):<10} | {average(scores['multi']):<9} | "
                     f"{average(overall):<7} | {average(seconds)}")
    summary = "\n".join(lines)
    SUMMARY.write_text(summary, encoding="utf-8")
    print("\n" + summary)