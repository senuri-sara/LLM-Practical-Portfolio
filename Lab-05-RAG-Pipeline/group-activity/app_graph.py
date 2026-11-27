"""
Streamlit demo: ask the same question with the Lab 05 RAG or the Graph RAG and compare.
Run with:  streamlit run app_graph.py
"""
import time

import streamlit as st

from graph_rag import graph_ask, graph      # also makes the Lab 05 files importable
from rag_pipeline import ask

st.set_page_config(page_title="University Knowledge Assistant – Graph RAG", layout="wide")
st.title("University Knowledge Assistant – Graph RAG")
st.caption(f"Knowledge graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} relationships "
           "from the NSBM Computing Awards Handbook")

mode = st.radio("Method", ["Compare both", "Graph RAG", "Lab 05 RAG"], horizontal=True)
question = st.text_input("Ask a question about the computing degrees",
                         placeholder="e.g. Is Artificial Intelligence core or elective in Software Engineering?")

if question:
    columns = st.columns(2) if mode == "Compare both" else [st.container()]
    jobs = []
    if mode in ("Compare both", "Lab 05 RAG"):
        jobs.append(("Lab 05 RAG", ask))
    if mode in ("Compare both", "Graph RAG"):
        jobs.append(("Graph RAG", graph_ask))

    for column, (label, function) in zip(columns, jobs):
        with column:
            st.subheader(label)
            with st.spinner("Thinking..."):
                start = time.perf_counter()
                result = function(question)
                seconds = time.perf_counter() - start
            st.write(result["answer"])
            st.caption(f"Answered in {seconds:.1f} s")
            if "facts" in result:
                with st.expander(f"Graph facts used ({len(result['facts'])})"):
                    st.json(result["entities"])
                    st.text("\n".join(result["facts"]))
            elif result.get("sources"):
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.write("- " + source)