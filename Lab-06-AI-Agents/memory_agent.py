import sqlite3
from datetime import datetime

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

DB_FILE = "agent_memory.db"
HISTORY_LIMIT = 20      # how many past messages the agent sees each turn

llm = ChatOllama(model="llama3.2:3b", temperature=0)


#SQLite memory
def get_conn():
    return sqlite3.connect(DB_FILE)


def init_db():
    with get_conn() as conn:
        # Short-term memory: the conversation itself
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,            -- 'user' or 'assistant'
            content TEXT NOT NULL,
            created_at TEXT NOT NULL)""")
        # Long-term memory: important facts about the user
        conn.execute("""CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL)""")


def save_message(role, content):
    with get_conn() as conn:
        conn.execute("INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
                     (role, content, datetime.now().isoformat(timespec="seconds")))


def load_history(limit=HISTORY_LIMIT):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    history = []
    for role, content in reversed(rows):
        history.append(HumanMessage(content=content) if role == "user"
                       else AIMessage(content=content))
    return history


def load_facts():
    with get_conn() as conn:
        return [r[0] for r in conn.execute("SELECT fact FROM facts ORDER BY id")]


def clear_memory():
    with get_conn() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM facts")


# Tools 
@tool
def calculator(expression: str) -> str:
    """Calculates a maths expression. Use this for any arithmetic.
    Example input: '25*4' or '(10+5)/3'."""
    # FIX: turn x / × into * (before, they were deleted, so 25x4 became 254)
    expression = expression.replace("×", "*").replace("x", "*").replace("÷", "/").replace(" ", "")
    allowed = "0123456789+-*/()."
    if not all(ch in allowed for ch in expression):
        return "Invalid expression. Only numbers and + - * / ( ) are allowed."
    try:
        # FIX: __builtins__ needs TWO underscores on each side
        return f"The result of {expression} is {eval(expression, {'__builtins__': {}})}"
    except Exception as e:
        return f"Could not calculate: {e}"


@tool
def save_memory(fact: str) -> str:
    """Saves an important fact about the user for later, such as their name,
    course, or preferences. Example input: "The user's name is Alex"."""
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO facts (fact, created_at) VALUES (?, ?)",
                         (fact.strip(), datetime.now().isoformat(timespec="seconds")))
        return f"Saved to memory: {fact}"
    except sqlite3.IntegrityError:
        return "I already remember that."


# Memory Search tool 
STOP_WORDS = {"what", "is", "my", "the", "who", "are", "you", "your", "did", "tell",
              "about", "do", "i", "me", "a", "an", "of", "user", "user's", "remember"}


@tool
def memory_search(query: str) -> str:
    """Searches facts the user told you before (name, course, preferences).
    Use this whenever the user asks about themselves or something they said
    earlier. Example input: 'name'."""
    # Turn the query into keywords, e.g. "my name" -> ["name"]
    words = [w.strip("?.!,").lower() for w in query.split()]
    keywords = [w for w in words if w and w not in STOP_WORDS] or words

    conditions = " OR ".join("LOWER(fact) LIKE ?" for _ in keywords)
    params = [f"%{k}%" for k in keywords]
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT fact FROM facts WHERE {conditions} ORDER BY id DESC", params
        ).fetchall()

    if not rows:
        return "No matching memories found."
    return "Memories found:\n" + "\n".join(f"- {r[0]}" for r in rows)


tools = [calculator, save_memory, memory_search]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

BASE_PROMPT = (
    "You are a helpful assistant with memory. Use the calculator tool for any maths. "
    "When the user tells you something personal or important about themselves "
    "(for example their name), use the save_memory tool, then reply briefly. "
    "When the user asks about themselves or something they told you before, "
    "you MUST use the memory_search tool first. If it finds nothing, say you don't know."
)


def build_system_prompt():
    # Facts are no longer pasted into the prompt, so the agent must
    # use the memory_search tool to retrieve them (as the lab asks).
    return BASE_PROMPT


#Agent loop
def run_agent(user_input):
    messages = [SystemMessage(content=build_system_prompt())]
    messages += load_history()                      # past conversation from SQLite
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

    # Store this exchange so the agent remembers it next time
    save_message("user", user_input)
    save_message("assistant", answer)
    return answer


# FIX: __name__ and __main__ need TWO underscores on each side
if __name__ == "__main__":
    init_db()
    print("Memory Agent ready! Commands: /memory (show facts), /clear (wipe memory), exit\n")
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