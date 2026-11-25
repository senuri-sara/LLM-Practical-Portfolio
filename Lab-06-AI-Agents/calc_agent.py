import subprocess   # opens Windows apps
import time         # waits for apps to open
import os           # files and folders
import platform     # system info
from datetime import datetime  # current date and time

import psutil                 # memory info
import pyautogui              # types keys automatically
import pygetwindow as gw      # finds and focuses windows

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Initialize LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)


# Open an app and bring it to the front
def open_app(exe_name, window_title):
    subprocess.Popen(exe_name)
    time.sleep(2)
    windows = gw.getWindowsWithTitle(window_title)
    if windows:
        try:
            windows[0].activate()
        except Exception:
            pass
        time.sleep(0.5)


# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """Opens the Windows Calculator and calculates a maths expression.
    Use this for any arithmetic. Example input: '25*4' or '(10+5)/3'."""
    # x, × and ÷ become * and /
    expression = expression.replace("×", "*").replace("x", "*").replace("÷", "/").replace(" ", "")

    # Only numbers and maths symbols allowed
    if not all(ch in "0123456789+-*/()." for ch in expression):
        return "Invalid expression. Only numbers and + - * / ( ) are allowed."

    try:
        result = eval(expression, {"__builtins__": {}})
    except Exception:
        return f"Could not calculate {expression}."

    # Show the calculation in the Calculator app
    open_app("calc.exe", "Calculator")
    pyautogui.press("esc")
    pyautogui.write(expression, interval=0.15)
    pyautogui.press("enter")

    return f"The result of {expression} is {result}"


# Tool 2: Date and time
@tool
def get_date_time() -> str:
    """Gets the current date and time. Use this when the user asks
    about today's date, the day, or the current time."""
    return datetime.now().strftime("Today is %A, %d %B %Y. The time is %I:%M %p.")


# Tool 3: Notepad
@tool
def write_in_notepad(text: str) -> str:
    """Opens Windows Notepad and types the given text into it.
    Use this when the user asks to open Notepad and write something."""
    open_app("notepad.exe", "Notepad")
    pyautogui.write(text, interval=0.05)
    return f"Opened Notepad and wrote: {text}"


# Tool 4: File search
@tool
def search_files(directory: str, pattern: str) -> str:
    """Searches a folder (and its subfolders) for files whose name or
    extension matches the pattern. Example: directory='D:\\MSC AI', pattern='.pdf'."""
    if not os.path.isdir(directory):
        return f"The folder '{directory}' does not exist."

    pattern = pattern.lower().replace("*", "")   # '*.pdf' -> '.pdf'
    found = []
    for folder, _, files in os.walk(directory):
        for name in files:
            if pattern in name.lower():
                found.append(os.path.join(folder, name))

    if not found:
        return f"No files matching '{pattern}' found in {directory}."
    return f"Found {len(found)} file(s):\n" + "\n".join(found[:20])


# Tool 5: File creation
@tool
def create_file(directory: str, filename: str, content: str) -> str:
    """Creates a text file with the given content in a folder.
    Example: directory='D:\\MSC AI', filename='test.txt', content='Hello World'."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created file {path} with content: {content}"


# Tool 6: System information
@tool
def get_system_info() -> str:
    """Gets this computer's specifications: operating system, processor
    and memory (RAM). Use this when the user asks about computer specs."""
    memory = psutil.virtual_memory()
    return (
        f"Operating system: {platform.system()} {platform.release()}\n"
        f"Processor: {platform.processor()}\n"
        f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n"
        f"Total RAM: {memory.total / (1024**3):.1f} GB\n"
        f"Available RAM: {memory.available / (1024**3):.1f} GB"
    )


# Register tools with LLM
tools = [calculator, get_date_time, write_in_notepad, search_files, create_file, get_system_info]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools on a Windows computer. "
    "Always use the right tool when the user asks for a calculation, the date or time, "
    "writing in Notepad, finding files, creating a file, or computer specifications. "
    "For calculations, give the calculator a plain expression like 25*4. "
    "After using a tool, give the user a short, clear final answer."
)


# Agent execution loop
def run_agent(user_input):
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]

    for _ in range(5):   # max 5 rounds
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool needed, so this is the final answer
        if not response.tool_calls:
            return response.content

        # Run each tool the model asked for
        for call in response.tool_calls:
            args = call["args"]
            # Llama 3.2 sometimes wraps the arguments inside "parameters"
            if isinstance(args.get("parameters"), dict):
                args = args["parameters"]

            print(f"  [Using tool: {call['name']} with {args}]")
            selected_tool = tools_by_name.get(call["name"])
            try:
                result = selected_tool.invoke(args) if selected_tool else "Unknown tool"
            except Exception as e:
                result = f"Tool error: {e}"

            # Send the result back to the model
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: too many steps."


# Run and test
if __name__ == "__main__":
    print("Calculator Agent ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input:
            print("Agent:", run_agent(user_input), "\n")