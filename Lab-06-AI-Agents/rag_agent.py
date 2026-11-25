from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# Reuse from Part B
from memory_agent import (llm, calculator, save_memory, memory_search, init_db, load_facts, clear_memory,
save_message, load_history)

from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# RAG Search tool 
# Opens the knowledge base built in Lab 05 (same collection and embedding model)
vectorstore = Chroma(
    collection_name="university_knowledge",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
    persist_directory=str(Path(__file__).resolve().parent / "chroma_db"),
)
@tool
def rag_search(query: str) -> str:
    """Searches the university knowledge base (the Computing Award Handbook).
    Use this for questions about degrees, courses, modules, credits,
    awards, grading or university rules. Example: 'credits needed for BSc'."""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in the knowledge base."

    results = []
    for d in docs:
        source = d.metadata.get("filename", "unknown")
        page = d.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        results.append(f"[Source: {source}{page_text}]\n{d.page_content}")
    return "\n\n".join(results)
# Register tools 
tools = [calculator, save_memory, memory_search, rag_search]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "You are a helpful university assistant with memory and a knowledge base. "
    "Choose the right tool:\n"
    "- calculator: for any maths.\n"
    "- save_memory: when the user tells you something personal (e.g. their name).\n"
    "- memory_search: when the user asks about themselves or what they told you before.\n"
    "- rag_search: for ANY question about the university, degrees, modules, credits, "
    "awards or rules. Answer ONLY from the rag_search results and mention the source page. "
    "If the results don't contain the answer, say you couldn't find it in the handbook."
)
#Agent loop
def run_agent(user_input):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages += load_history()
    messages.append(HumanMessage(content=user_input))

    answer = "Stopped: too many steps."
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            answer = response.content
            break

        for call in response.tool_calls:
            print(f"  [Using tool: {call['name']} with {call['args']}]")
            selected_tool = tools_by_name.get(call["name"])
            try:
                result = selected_tool.invoke(call["args"]) if selected_tool else "Unknown tool"
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    save_message("user", user_input)
    save_message("assistant", answer)
    return answer


# Run and test 
if __name__ == "__main__":
    init_db()
    print("RAG Agent ready! Commands: /memory, /clear, exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input == "":
            continue
        if user_input == "/memory":
            print("Saved facts:", load_facts() or "none", "\n")
            continue
        if user_input == "/clear":
            clear_memory()
            print("Memory cleared.\n")
            continue
        print("Agent:", run_agent(user_input), "\n")