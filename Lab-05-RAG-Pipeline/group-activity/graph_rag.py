"""
Graph RAG: answers questions using the knowledge graph plus the Lab 05 text retrieval.

1. Entity linking  - find programmes, modules, years and types in the question
2. Graph traversal - follow the graph links to collect exact facts
3. Aggregation     - counts, credit totals and common modules are worked out in Python
4. Generation      - Llama 3.2 answers from the graph facts and the retrieved text
"""
import json
import re
import sys
from pathlib import Path

from networkx.readwrite import json_graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))            # find the Lab 05 files (rag_pipeline.py)
from rag_pipeline import retrieve, format_context, clean_answer, NOT_FOUND   # noqa: E402

llm = ChatOllama(model="llama3.2:3b", temperature=0, num_ctx=8192)

# Words people use for each programme
PROGRAMME_ALIASES = {
    "BSc in Management Information Systems (Special)": ["management information systems", "mis"],
    "BSc (Hons) in Software Engineering": ["software engineering"],
    "BSc (Hons) in Computer Science": ["computer science"],
    "BSc (Hons) in Computer Networks": ["computer networks", "networking degree"],
    "BSc in Multimedia": ["multimedia"],
}
YEAR_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "final": 4}
ALL_WORDS = ["all programmes", "all programs", "all degrees", "every programme", "every program",
             "every degree", "each programme", "each program", "each degree", "all the degrees"]


def load_graph():
    with open(BASE_DIR / "graph.json", encoding="utf-8") as f:
        return json_graph.node_link_graph(json.load(f), multigraph=True, directed=True, edges="links")


graph = load_graph()
MODULES = {n: d for n, d in graph.nodes(data=True) if d["kind"] == "Module"}


# ---------- 1. Entity linking ----------
def link_entities(question):
    q = " " + question.lower() + " "
    found = {"programmes": [], "modules": [], "years": [], "types": []}

    for programme, aliases in PROGRAMME_ALIASES.items():
        if any(re.search(r"\b" + re.escape(a) + r"\b", q) for a in aliases):
            found["programmes"].append(programme)
    if any(w in q for w in ALL_WORDS):
        found["programmes"] = list(PROGRAMME_ALIASES)

    for code, data in MODULES.items():
        code_pattern = code.lower().replace(" ", r"\s?").replace(".", r"\.")
        if re.search(r"\b" + code_pattern, q) or data["name"].lower() in q:
            found["modules"].append(code)

    for number in re.findall(r"year\s*0?([1-4])|([1-4])(?:st|nd|rd|th)\s+year", q):
        found["years"].append(int(number[0] or number[1]))
    for word, number in YEAR_WORDS.items():
        if re.search(r"\b" + word + r"[- ]year", q):
            found["years"].append(number)

    if re.search(r"\b(core|compulsory)\b", q):
        found["types"].append("Core")
    if re.search(r"\b(elective|electives|optional)\b", q):
        found["types"].append("Elective")

    found["years"] = sorted(set(found["years"]))
    return found


# ---------- 2 and 3. Graph traversal and aggregation ----------
def programme_modules(programme, years=None, types=None):
    rows = []
    for _, code, d in graph.out_edges(programme, data=True):
        if d["relation"] != "HAS_MODULE":
            continue
        if years and d["year"] not in years:
            continue
        if types and d["type"] not in types:
            continue
        rows.append((d["year"], code, d["name_here"], graph.nodes[code]["credits"], d["type"]))
    return sorted(rows)


def programme_info(p):
    info = graph.nodes[p]
    careers = [o for _, o, d in graph.out_edges(p, data=True) if d["relation"] == "LEADS_TO"]
    return (f"{p}: duration {info['duration']}; academic progression {info['progression']}; "
            f"career prospects: {', '.join(careers)}.")


def module_names(codes):
    return ", ".join(f"{c} {graph.nodes[c]['name']}" for c in sorted(codes)) or "none"


def graph_facts(entities):
    facts = []
    years = entities["years"]
    types = entities["types"] if len(entities["types"]) == 1 else []   # "core or elective" = no filter
    module_sets = {}

    # Hop 1: programmes named in the question
    for p in entities["programmes"]:
        facts.append(programme_info(p))
        # For a question about specific modules, only list a programme's modules if a year is given
        if entities["modules"] and not years:
            continue
        rows = programme_modules(p, years, types)
        module_sets[p] = {r[1] for r in rows}
        for year, code, name, credits, mtype in rows:
            facts.append(f"{p} | Year {year} | {code} {name} | {credits} credits | {mtype}")
        scope = (f"Year {', '.join(map(str, years))}" if years else "all years") + \
                (f", {types[0]} only" if types else "")
        facts.append(f"CALCULATED: {p} ({scope}) has {len(rows)} modules worth "
                     f"{sum(r[3] for r in rows)} credits in total "
                     f"({sum(r[4] == 'Core' for r in rows)} core, {sum(r[4] == 'Elective' for r in rows)} elective).")

    # Comparisons between programmes
    if len(module_sets) > 1:
        common = set.intersection(*module_sets.values())
        facts.append(f"CALCULATED: modules common to {', '.join(module_sets)}: "
                     f"{module_names(common)} ({len(common)} modules).")
    if len(module_sets) == 2:
        (a, set_a), (b, set_b) = module_sets.items()
        facts.append(f"CALCULATED: only in {a}: {module_names(set_a - set_b)}.")
        facts.append(f"CALCULATED: only in {b}: {module_names(set_b - set_a)}.")

    # Modules named in the question, and hop 2: the programmes that offer them
    hop2 = []
    for code in entities["modules"]:
        data = graph.nodes[code]
        offered = []
        for p, _, d in graph.in_edges(code, data=True):
            if d["relation"] == "HAS_MODULE":
                offered.append(f"{p} (Year {d['year']}, {d['type']})")
                if p not in entities["programmes"] and p not in hop2:
                    hop2.append(p)
        facts.append(f"{code} {data['name']} ({data['credits']} credits) is offered in: {'; '.join(offered)}.")
    if len(hop2) <= 3:
        facts += [programme_info(p) for p in hop2]

    return facts


# ---------- 4. Generation ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a university assistant for the NSBM Faculty of Computing. Answer using the GRAPH FACTS "
     "first: they are exact data from the handbook tables. Lines starting with CALCULATED are already "
     "worked out, so use those numbers and do not recalculate. Use the TEXT CONTEXT for descriptions. "
     f"If neither contains the answer, say: \"{NOT_FOUND}\" Do not make up information.\n\n"
     "GRAPH FACTS:\n{facts}\n\nTEXT CONTEXT:\n{context}"),
    ("human", "{question}"),
])


def graph_ask(question):
    entities = link_entities(question)
    facts = graph_facts(entities)
    docs, _ = retrieve(question)
    response = (answer_prompt | llm).invoke({
        "facts": "\n".join(facts) if facts else "No matching graph facts.",
        "context": format_context(docs[:4]),
        "question": question,
    })
    return {"answer": clean_answer(response.content), "entities": entities, "facts": facts}


if __name__ == "__main__":
    print("Graph RAG ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if question:
            result = graph_ask(question)
            print(f"  [Entities: {result['entities']}]")
            print(f"  [Graph facts used: {len(result['facts'])}]")
            print("Agent:", result["answer"], "\n")