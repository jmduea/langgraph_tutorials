"""Adding HITLC (Human In The Loop Controls) to LangGraph chatbot."""

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

# Web search tool
from langchain_tavily import TavilySearch

# Chat Memory management
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition

# for HITLC
from langgraph.types import Command
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Request assistance from a human using direct terminal input."""
    user_confirm = input(
        f"Is the information correct? Name: {name}, Birthday: {birthday} (y/n): "
    )
    if user_confirm.strip().lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Information verified as correct."
    else:
        verified_name = input(f"Please provide the correct name (current: {name}): ")
        verified_birthday = input(
            f"Please provide the correct birthday (current: {birthday}): "
        )
        response = "Information updated with user corrections."

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


def chatbot(state: State):
    # Update to use the LLM with tools
    message = llm_with_tools.invoke(state["messages"])
    # disable parallel tool calling to avoid repeated tool calls
    # assert len(message.tool_calls) <= 1
    return {"messages": [message]}


def stream_graph_updates(user_input: str, config):
    # Passing config for conversation history management
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            print(f"{event.keys()}")
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    graph_builder = StateGraph(State)

    # Use local model hosted with Ollama
    # Ensure you have Ollama installed and the model is available
    # qwen3:8b for tool use
    llm = init_chat_model("ollama:qwen3:8b")

    tool = TavilySearch(max_results=2)
    tools = [tool, human_assistance]

    llm_with_tools = llm.bind_tools(tools)

    # In memory checkpointer, should replace with SqliteSaver or other persistent storage in production
    memory = MemorySaver()

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "2"}}

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, config)
        except Exception as e:
            print(f"An error occurred: {e}")
            if "[WinError 10061]" in str(e):
                print("Connection error. Please ensure the Ollama server is running.")
                exit(1)
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, config)
            break
