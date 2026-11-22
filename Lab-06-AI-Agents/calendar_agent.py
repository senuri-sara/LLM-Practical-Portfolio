import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

# Reuse the Google Calendar tools
from calendar_tools import create_event, get_today_schedule, get_calendar_service

#Initialize LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Register tools with LLM
tools = [create_event, get_today_schedule]
tools_by_name = {t.name: t for t in tools}   # {"create_event": create_event, ...}
llm_with_tools = llm.bind_tools(tools)

#Agent execution loop
SYSTEM_PROMPT = (
    "You are a helpful calendar assistant for the user's Google Calendar.\n"
    "- create_event: when the user wants to add an event or meeting. "
    "Convert dates to YYYY-MM-DD and times to 24-hour HH:MM.\n"
    "- get_today_schedule: when the user asks what's on today.\n"
    f"Today's date is {datetime.datetime.now():%A, %Y-%m-%d}. "
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

        # No tool requested -> this is the final answer
        if not response.tool_calls:
            return response.content

        # The AI asked for a tool -> run it
        for call in response.tool_calls:
            print(f"  [Using tool: {call['name']} with {call['args']}]")
            selected_tool = tools_by_name.get(call["name"])
            try:
                result = selected_tool.invoke(call["args"]) if selected_tool else "Unknown tool"
            except Exception as e:
                result = f"Tool error: {e}"

            # Send the tool's result back to the AI
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: too many steps."


#Run and test  (must always be LAST in the file)
if __name__ == "__main__":
    get_calendar_service()          # sign in first (opens browser only the first time)
    print("Calendar Agent ready! Type 'exit' to quit.")
    print("Try: What's on my schedule today?  |  Create an event called Study Group tomorrow at 3pm\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input == "":
            continue

        answer = run_agent(user_input)
        print("Agent:", answer, "\n")