import sqlite3
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

DB_FILE = "agent_memory.db"
HISTORY_LIMIT = 20      # how many past messages the agent sees each turn

# Initialize LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)


# SQLite memory
def get_conn():
    return sqlite3.connect(DB_FILE)


def init_db():
    with get_conn() as conn:
        # Short-term memory: the conversation
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL)""")
        # Long-term memory: facts about the user
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
        history.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    return history


def load_facts():
    with get_conn() as conn:
        return [r[0] for r in conn.execute("SELECT fact FROM facts ORDER BY id")]


def clear_memory():
    with get_conn() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM facts")


# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """Calculates a maths expression. Use this for any arithmetic.
    Example input: '25*4' or '(10+5)/3'."""
    expression = expression.replace("×", "*").replace("x", "*").replace("÷", "/").replace(" ", "")
    if not all(ch in "0123456789+-*/()." for ch in expression):
        return "Invalid expression. Only numbers and + - * / ( ) are allowed."
    try:
        return f"The result of {expression} is {eval(expression, {'__builtins__': {}})}"
    except Exception:
        return f"Could not calculate {expression}."


# Tool 2: Save memory
@tool
def save_memory(fact: str) -> str:
    """Saves ONE important fact about the user as a full sentence, such as
    "The user's name is Alex" or "The user studies Data Science".
    Call it once for each fact."""
    fact = fact.strip()
    # Facts like "The user" are too short to be useful
    if len(fact.split()) < 4:
        return "Fact too short. Save a full sentence, for example: The user's name is Alex."
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO facts (fact, created_at) VALUES (?, ?)",
                         (fact, datetime.now().isoformat(timespec="seconds")))
        return f"Saved to memory: {fact}"
    except sqlite3.IntegrityError:
        return "I already remember that."


# Tool 3: Memory search
STOP_WORDS = {"what", "is", "my", "the", "who", "are", "you", "your", "did", "tell",
              "about", "do", "i", "me", "a", "an", "of", "user", "user's", "remember"}


@tool
def memory_search(query: str) -> str:
    """Searches facts the user told you before (name, course, preferences).
    Use this whenever the user asks about themselves or something they said
    earlier. Example input: 'name'."""
    # "my name" -> ["name"]
    words = [w.strip("?.!,").lower() for w in query.split()]
    keywords = [w for w in words if w and w not in STOP_WORDS] or words

    conditions = " OR ".join("LOWER(fact) LIKE ?" for _ in keywords)
    params = [f"%{k}%" for k in keywords]
    with get_conn() as conn:
        rows = [r[0] for r in conn.execute(
            f"SELECT fact FROM facts WHERE {conditions} ORDER BY id DESC", params)]

    # Keywords can miss (e.g. "name" vs "The user is Senuri"),
    # so if nothing matches, give the model all saved facts
    if not rows:
        rows = load_facts()
    if not rows:
        return "No memories saved yet."
    return "Memories found:\n" + "\n".join(f"- {f}" for f in rows)


# Register tools with LLM
tools = [calculator, save_memory, memory_search]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

BASE_PROMPT = (
    "You are a helpful assistant with memory. Use the calculator tool for any maths. "
    "When the user tells you something personal or important about themselves "
    "(for example their name or course), save each fact separately with the save_memory tool "
    "as a full sentence such as \"The user's name is Alex\", then reply briefly. "
    "When the user asks about themselves or something they told you before, "
    "you MUST use the memory_search tool first and answer from the memories it finds. "
    "If it says no memories are saved, say you don't know."
)


# Fix common argument mistakes from the small model
def fix_args(selected_tool, args):
    # {'parameters': {...}} -> {...}
    if isinstance(args.get("parameters"), dict):
        args = args["parameters"]
    # Wrong name for a one-argument tool, e.g. {'query': ...} instead of {'fact': ...}
    expected = list(selected_tool.args)
    if len(expected) == 1 and len(args) == 1 and expected[0] not in args:
        args = {expected[0]: list(args.values())[0]}
    return args


# Agent loop
def run_agent(user_input):
    messages = [SystemMessage(content=BASE_PROMPT)]
    messages += load_history()                      # past conversation from SQLite
    messages.append(HumanMessage(content=user_input))

    answer = "Stopped: too many steps."
    for _ in range(5):   # max 5 rounds
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool needed, so this is the final answer
        if not response.tool_calls:
            answer = response.content
            break

        # Run each tool the model asked for
        for call in response.tool_calls:
            selected_tool = tools_by_name.get(call["name"])
            if selected_tool is None:
                result = "Unknown tool"
            else:
                args = fix_args(selected_tool, call["args"])
                print(f"  [Using tool: {call['name']} with {args}]")
                try:
                    result = selected_tool.invoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Save this exchange so the agent remembers it next time
    save_message("user", user_input)
    save_message("assistant", answer)
    return answer


# Run and test
if __name__ == "__main__":
    init_db()
    print("Memory Agent ready! Commands: /memory (show facts), /clear (wipe memory), exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input == "/memory":
            print("Saved facts:", load_facts() or "none", "\n")
            continue
        if user_input == "/clear":
            clear_memory()
            print("Memory cleared.\n")
            continue
        if user_input:
            print("Agent:", run_agent(user_input), "\n")