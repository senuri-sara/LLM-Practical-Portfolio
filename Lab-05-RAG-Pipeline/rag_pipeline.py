"""
RAG pipeline: retrieve relevant chunks from ChromaDB and answer with Llama 3.2.
Improvements: query rewriting + MMR search + balanced grounding prompt.
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from ingest import get_vectorstore

LLM_MODEL = "llama3.2:3b"
TOP_K = 6          # chunks returned per search
FETCH_K = 20       # candidates considered by MMR before choosing the most varied ones
NOT_FOUND = "I could not find this information in the uploaded documents."

# temperature=0 keeps answers factual and consistent
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# Prompt 1: turn the user's question into a better search query
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's question as a short search query for finding the answer in a "
     "university handbook. Add useful related keywords (for example: programme aims, "
     "objectives, description, modules, rules, requirements). Return only the search query."),
    ("human", "{question}")
])

# Prompt 2: answer using only the retrieved context
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful university assistant. Answer the question using ONLY the context below, "
     "which comes from university documents. If the context contains related information that "
     "partly answers the question (for example, a programme description or aims), use it and "
     "explain that your answer is based on that information. Only if the context contains "
     f"nothing relevant, say: \"{NOT_FOUND}\" Do not make up information.\n\nContext:\n{{context}}"),
    ("human", "{question}")
])


def rewrite_query(question):
    """Ask the LLM for a better search query; fall back to the original question."""
    try:
        query = (rewrite_prompt | llm).invoke({"question": question}).content.strip()
    except Exception:
        return question

    # If the rewrite is empty, too long or a refusal, use the original question instead
    refusal_words = ["can't", "cannot", "sorry", "unable", "not able"]
    if not query or len(query.split()) > 20 or any(w in query.lower() for w in refusal_words):
        return question
    return query

def retrieve(question, filenames=None, k=TOP_K):
    """Search with the original and rewritten queries using MMR, then remove duplicates."""
    vectorstore = get_vectorstore()
    search_filter = {"filename": {"$in": list(filenames)}} if filenames else None

    search_query = rewrite_query(question)
    docs, seen = [], set()

    for query in [question, search_query]:
        results = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=FETCH_K, filter=search_filter
        )
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)

    return docs[:k + 2], search_query


def format_context(docs):
    """Join the retrieved chunks, labelling each with its file and page."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", 0) + 1
        parts.append(f"[Source {i}: {filename}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def clean_answer(text):
    """Remove a stray 'assistant' label that Llama 3.2 sometimes adds."""
    text = text.strip()
    if text.lower().startswith("assistant"):
        text = text[len("assistant"):].strip()
    return text


def ask(question, filenames=None):
    """Answer a question and return the answer, its sources and the search query used."""
    docs, search_query = retrieve(question, filenames)

    if not docs:
        return {"answer": "No documents found. Please upload a PDF first.",
                "sources": [], "search_query": search_query}

    response = (answer_prompt | llm).invoke(
        {"context": format_context(docs), "question": question}
    )
    answer = clean_answer(response.content)

    # Don't show sources when the answer was not found
    if NOT_FOUND.lower() in answer.lower():
        sources = []
    else:
        sources = sorted({
            f"{d.metadata.get('filename', 'unknown')} – page {d.metadata.get('page', 0) + 1}"
            for d in docs
        })

    return {"answer": answer, "sources": sources, "search_query": search_query}


def list_indexed_files():
    """Return the names of all PDFs stored in the vector database."""
    data = get_vectorstore().get(include=["metadatas"])
    return sorted({m.get("filename", "unknown") for m in data["metadatas"]})


if __name__ == "__main__":
    print("Indexed documents:", list_indexed_files())

    while True:
        question = input("\nAsk a university question (or exit): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        result = ask(question)
        print("\nSearch query used:", result["search_query"])
        print("\nAnswer:", result["answer"])
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(" -", source)