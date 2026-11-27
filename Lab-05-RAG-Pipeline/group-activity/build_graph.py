"""
Builds a knowledge graph from the NSBM Computing Awards Handbook.

Entities:   Faculty, Programme, Module, Career
Relations:  Faculty  -OFFERS->      Programme
            Programme -HAS_MODULE-> Module   (year, type, credits)
            Programme -LEADS_TO->   Career
Output:     graph.json (the graph) and triples.csv (for checking by hand)
"""
import csv
import json
import re
import sys
from pathlib import Path

import networkx as nx
import pymupdf
from networkx.readwrite import json_graph

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR.parent / "data" / "COMPUTING AWARD HANDBOOK_MOHE (2).pdf"   # Lab 05 data folder

# Programme headings as they appear in the handbook, with a short name
PROGRAMMES = {
    "Bachelor of Science in Management Information Systems (Special)": "BSc in Management Information Systems (Special)",
    "Bachelor of Science (Honours) in Software Engineering": "BSc (Hons) in Software Engineering",
    "Bachelor of Science (Honours) in Computer Science": "BSc (Hons) in Computer Science",
    "Bachelor of Science (Honours) in Computer Networks": "BSc (Hons) in Computer Networks",
    "Bachelor of Science in Multimedia": "BSc in Multimedia",
}
FACULTY = "Faculty of Computing"
MODULE_ROW = re.compile(r"\b([A-Z]{2}\s?\d{3,4}(?:\.\d)?)\s+(.+?)\s+(\d)\s+(Core|Elective)\b")


def clean(text):
    return " ".join(str(text or "").replace("#", "").split())


def read_field(text, label, next_labels):
    """Reads the text after a label such as DURATION, up to the next label."""
    pattern = label + r":?\s*(.+?)\s*(?=" + "|".join(next_labels) + r"|$)"
    match = re.search(pattern, text, re.S)
    return clean(match.group(1)) if match else ""


def extract(pdf_path):
    """Returns programme details and module rows from the handbook."""
    doc = pymupdf.open(pdf_path)
    programmes, rows = {}, []
    current, year = None, 0

    for page in doc:
        text = page.get_text()

        # A programme overview page starts a new programme
        for heading, short in PROGRAMMES.items():
            if heading in text and "AWARD" in text:
                current, year = short, 0
                programmes[short] = {
                    "duration": read_field(text, "DURATION", ["ACADEMIC"]),
                    "progression": read_field(text, r"ACADEMIC PROGRESSION", ["CAREER"]),
                    "careers": read_field(text, r"CAREER PROSPECTS", ["COMPUTING AWARDS", r"\Z"]),
                }

        # Each module table is the next year of the current programme
        for table in page.find_tables().tables:
            year += 1
            for row in table.extract():
                # Join the cells and read: code, name, credits, type
                match = MODULE_ROW.search(" ".join(clean(c) for c in row if c))
                if match:
                    code, name, credits, mtype = match.groups()
                    rows.append({
                        "programme": current, "year": year,
                        "code": code, "name": name,
                        "credits": int(credits), "type": mtype,
                    })
    return programmes, rows


def build_graph(programmes, rows):
    graph = nx.MultiDiGraph()
    graph.add_node(FACULTY, kind="Faculty")
    conflicts = []

    for name, info in programmes.items():
        graph.add_node(name, kind="Programme", duration=info["duration"],
                       progression=info["progression"])
        graph.add_edge(FACULTY, name, relation="OFFERS")
        for career in info["careers"].rstrip(".").split(","):
            career = clean(career)
            if career:
                graph.add_node(career, kind="Career")
                graph.add_edge(name, career, relation="LEADS_TO")

    for r in rows:
        node = r["code"]
        if node in graph and graph.nodes[node]["name"] != r["name"]:
            conflicts.append(f"{node}: '{graph.nodes[node]['name']}' vs '{r['name']}' ({r['programme']}, Year {r['year']})")
        else:
            graph.add_node(node, kind="Module", name=r["name"], credits=r["credits"])
        graph.add_edge(r["programme"], node, relation="HAS_MODULE",
                       year=r["year"], type=r["type"], name_here=r["name"])
    return graph, conflicts


if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_PATH
    programmes, rows = extract(pdf)
    graph, conflicts = build_graph(programmes, rows)

    # Save the graph
    with open(BASE_DIR / "graph.json", "w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(graph, edges="links"), f, indent=1)

    # Save every fact as a triple so it can be checked against the PDF
    with open(BASE_DIR / "triples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "relation", "object", "year", "type", "credits"])
        for s, o, d in graph.edges(data=True):
            label = f"{o} {graph.nodes[o].get('name', '')}".strip()
            writer.writerow([s, d["relation"], label, d.get("year", ""), d.get("type", ""),
                             graph.nodes[o].get("credits", "")])

    # Summary
    kinds = {}
    for _, d in graph.nodes(data=True):
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"Nodes: {graph.number_of_nodes()} {kinds}")
    print(f"Edges: {graph.number_of_edges()}")
    for name in programmes:
        years = {}
        for _, _, d in graph.out_edges(name, data=True):
            if d["relation"] == "HAS_MODULE":
                years[d["year"]] = years.get(d["year"], 0) + 1
        print(f"  {name}: {programmes[name]['duration']}, modules per year {dict(sorted(years.items()))}")
    print("Conflicts found in the handbook:" if conflicts else "No conflicts found.")
    for c in conflicts:
        print("  -", c)