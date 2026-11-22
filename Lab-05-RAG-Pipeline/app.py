"""
University AI Assistant - Streamlit chat app with multi-PDF upload.
"""
import streamlit as st

from ingest import ingest_pdf
from rag_pipeline import ask, list_indexed_files

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")

st.title("🎓 University AI Assistant")
st.caption("Ask questions based on university documents")

# ---------- Sidebar: upload and manage PDFs ----------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Add to knowledge base", disabled=not uploaded_files):
        for file in uploaded_files:
            with st.spinner(f"Indexing {file.name}..."):
                try:
                    message = ingest_pdf(file)
                    if message.startswith("Already indexed"):
                        st.info(message)
                    else:
                        st.success(message)
                except Exception as e:
                    st.error(f"{file.name}: {e}")

    indexed_files = list_indexed_files()

    st.subheader("Indexed documents")
    if indexed_files:
        for name in indexed_files:
            st.write("•", name)
    else:
        st.write("No documents yet. Upload a PDF to start.")

    selected_files = st.multiselect(
        "Search only in (optional)",
        indexed_files,
        help="Leave empty to search all documents."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write("•", source)

# ---------- New question ----------
if question := st.chat_input("Ask your university question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                result = ask(question, filenames=selected_files or None)
            answer, sources = result["answer"], result["sources"]
        except Exception:
            answer, sources = "Sorry, something went wrong. Please check that Ollama is running.", []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write("•", source)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})