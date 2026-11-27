"""
Draws part of the knowledge graph: one programme, its modules coloured by year,
and the other programmes that share those modules.
"""
import json
import math
import sys

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph

from build_graph import BASE_DIR

PROGRAMME = sys.argv[1] if len(sys.argv) > 1 else "BSc (Hons) in Computer Science"
YEAR_COLOURS = {1: "#9ecae1", 2: "#6baed6", 3: "#3182bd", 4: "#08519c"}

with open(BASE_DIR / "graph.json", encoding="utf-8") as f:
    graph = json_graph.node_link_graph(json.load(f), multigraph=True, directed=True, edges="links")

# Sub-graph: the programme, its modules, and other programmes offering the same modules
sub = nx.Graph()
for _, code, d in graph.out_edges(PROGRAMME, data=True):
    if d["relation"] == "HAS_MODULE":
        sub.add_edge(PROGRAMME, code, style="solid" if d["type"] == "Core" else "dashed")
        sub.nodes[code]["year"] = d["year"]
        for other, _, d2 in graph.in_edges(code, data=True):
            if other != PROGRAMME and d2["relation"] == "HAS_MODULE":
                sub.add_edge(other, code, style="dotted")

programmes = [n for n in sub if graph.nodes[n]["kind"] == "Programme"]
modules = [n for n in sub if n not in programmes]

# Layout: programme in the centre, modules in rings by year, other programmes in the corners
pos = {PROGRAMME: (0, 0)}
for year in YEAR_COLOURS:
    ring = sorted(m for m in modules if sub.nodes[m]["year"] == year)
    for i, m in enumerate(ring):
        angle = 2 * math.pi * i / max(len(ring), 1) + year * 0.35
        pos[m] = (year * math.cos(angle), year * math.sin(angle))
corners = [(-5.2, 4.6), (5.2, 4.6), (5.2, -4.6), (-5.2, -4.6)]
for p, xy in zip([p for p in programmes if p != PROGRAMME], corners):
    pos[p] = xy

plt.figure(figsize=(15, 13))
for style in ["solid", "dashed", "dotted"]:
    edges = [(a, b) for a, b, d in sub.edges(data=True) if d["style"] == style]
    nx.draw_networkx_edges(sub, pos, edgelist=edges, style=style, alpha=0.2 if style == "dotted" else 0.75)
nx.draw_networkx_nodes(sub, pos, nodelist=programmes, node_color="#fd8d3c", node_size=5200, node_shape="s")
nx.draw_networkx_nodes(sub, pos, nodelist=modules, node_size=900,
                       node_color=[YEAR_COLOURS[sub.nodes[m]["year"]] for m in modules])
nx.draw_networkx_labels(sub, pos, labels={p: p.replace(" in ", "\n") for p in programmes}, font_size=8)
nx.draw_networkx_labels(sub, pos, labels={m: m for m in modules}, font_size=7, font_color="white")

for year, colour in YEAR_COLOURS.items():
    plt.scatter([], [], color=colour, s=120, label=f"Year {year} module")
plt.plot([], [], "k-", label="Core")
plt.plot([], [], "k--", label="Elective")
plt.plot([], [], "k:", label="Shared with another programme")
plt.legend(loc="lower left", fontsize=9)
plt.title(f"Knowledge graph: {PROGRAMME} ({len(modules)} modules, {len(programmes) - 1} related programmes)")
plt.axis("off")
plt.tight_layout()

out = BASE_DIR / "results" / "knowledge_graph.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=130)
print(f"Saved {out}")