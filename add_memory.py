from typing import Annotated

from langchain.chat_models import init_chat_model

# Web search tool
from langchain_tavily import TavilySearch

# Chat Memory management
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    # Update to use the LLM with tools
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def stream_graph_updates(user_input: str, config):
    # Passing config for conversation history management
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    graph_builder = StateGraph(State)

    # Use local model hosted with Ollama
    # Ensure you have Ollama installed and the model is available
    # qwen3:8b for tool use
    llm = init_chat_model("ollama:qwen3:8b")

    tool = TavilySearch(max_results=2)
    tools = [tool]

    llm_with_tools = llm.bind_tools(tools)

    # In memory checkpointer, should replace with SqliteSaver or other persistent storage in production
    memory = MemorySaver()

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
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
        except:
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input, config)
            break
