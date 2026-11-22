import subprocess   # opens Windows apps (like calc.exe)
import time         # lets us wait for Calculator to open

import pyautogui              # types keys automatically
import pygetwindow as gw      # finds and focuses windows

from langchain_ollama import ChatOllama          # talks to Ollama
from langchain_core.tools import tool            # turns a function into a tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage # chat messages

import os                    # files and folders
import platform              # operating system and processor info
from datetime import datetime  # current date and time
import psutil                # memory info

#Initialize LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)

#Create tool for calculator
@tool
def calculator(expression: str) -> str:
    """Opens the Windows Calculator and calculates a maths expression.
    Use this for any arithmetic. Example input: '25*4' or '(10+5)/3'."""

    # Clean up the expression (AI or user might write x, × or ÷)
    expression = expression.replace("×", "*").replace("x", "*").replace("÷", "/").replace(" ", "")

    # Safety check: only allow numbers and maths symbols
    allowed = "0123456789+-*/()."
    if not all(ch in allowed for ch in expression):
        return "Invalid expression. Only numbers and + - * / ( ) are allowed."

    # Work out the answer in Python
    result = eval(expression, {"__builtins__": {}})

    # Open Calculator and type the sum so we can see it happen
    subprocess.Popen("calc.exe")
    time.sleep(2)                                  # wait for Calculator to open

    windows = gw.getWindowsWithTitle("Calculator")
    if windows:
        try:
            windows[0].activate()                  # bring Calculator to the front
        except Exception:
            pass
        time.sleep(0.5)

    pyautogui.press("esc")                         # clear old calculation
    pyautogui.write(expression, interval=0.15)     # type e.g. 25*4
    pyautogui.press("enter")                       # press =

    return f"The result of {expression} is {result}"

#Expand the calculator with new  tools ----------

@tool
def get_date_time() -> str:
    """Gets the current date and time. Use this when the user asks
    about today's date, the day, or the current time."""
    now = datetime.now()
    return now.strftime("Today is %A, %d %B %Y. The time is %I:%M %p.")


@tool
def write_in_notepad(text: str) -> str:
    """Opens Windows Notepad and types the given text into it.
    Use this when the user asks to open Notepad and write something."""
    subprocess.Popen("notepad.exe")
    time.sleep(2)                                   # wait for Notepad to open

    windows = gw.getWindowsWithTitle("Notepad")
    if windows:
        try:
            windows[0].activate()
        except Exception:
            pass
        time.sleep(0.5)

    pyautogui.write(text, interval=0.05)
    return f"Opened Notepad and wrote: {text}"


@tool
def search_files(directory: str, pattern: str) -> str:
    """Searches a folder (and its subfolders) for files whose name or
    extension matches the pattern. Example: directory='D:\\MSC AI', pattern='.pdf'."""
    if not os.path.isdir(directory):
        return f"The folder '{directory}' does not exist."

    pattern = pattern.lower().replace("*", "")      # '*.pdf' -> '.pdf'
    found = []
    for folder, subfolders, files in os.walk(directory):
        for name in files:
            if pattern in name.lower():
                found.append(os.path.join(folder, name))

    if not found:
        return f"No files matching '{pattern}' found in {directory}."
    return f"Found {len(found)} file(s):\n" + "\n".join(found[:20])


@tool
def create_file(directory: str, filename: str, content: str) -> str:
    """Creates a text file with the given content in a folder.
    Example: directory='D:\\MSC AI', filename='test.txt', content='Hello World'."""
    os.makedirs(directory, exist_ok=True)           # create the folder if missing
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created file {path} with content: {content}"


@tool
def get_system_info() -> str:
    """Gets this computer's specifications: operating system, processor
    and memory (RAM). Use this when the user asks about computer specs."""
    memory = psutil.virtual_memory()
    return (
        f"Operating system: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n"
        f"Total RAM: {memory.total / (1024**3):.1f} GB\n"
        f"Available RAM: {memory.available / (1024**3):.1f} GB"
    )

# Register tool with LLM
tools = [calculator, get_date_time, write_in_notepad, search_files, create_file, get_system_info]
tools_by_name = {t.name: t for t in tools}   # {"calculator": calculator}
llm_with_tools = llm.bind_tools(tools)

#Agent execution loop
SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools on a Windows computer. "
    "Always use the right tool when the user asks for a calculation, the date or time, "
    "writing in Notepad, finding files, creating a file, or computer specifications. "
    "For calculations, give the calculator a plain expression like 25*4. "
    "After using a tool, give the user a short, clear final answer."
)


def run_agent(user_input):
    # The conversation so far
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    for step in range(5):                       # safety limit: max 5 rounds
        response = llm_with_tools.invoke(messages)
        messages.append(response)               # remember what the AI said

        # No tool requested → this is the final answer
        if not response.tool_calls:
            return response.content

        # The AI asked for a tool → run it
        for call in response.tool_calls:
            print(f"  [Using tool: {call['name']} with {call['args']}]")
            selected_tool = tools_by_name[call["name"]]
            result = selected_tool.invoke(call["args"])

            # Send the tool's result back to the AI
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: too many steps."


#Run and test  (must always be LAST in the file)
if __name__ == "__main__":
    print("Calculator Agent ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input == "":
            continue

        answer = run_agent(user_input)
        print("Agent:", answer, "\n")