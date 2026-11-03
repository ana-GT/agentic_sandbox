from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


agent = create_agent("ollama:gemma4:e2b", tools=[get_weather])


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text or token.tool_call_chunks:
      if token.text:
          print(token.text, end="|")
      if token.tool_call_chunks:
          print(token.tool_call_chunks, end="/")
      # N.B. all content is available through token.content_blocks


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n* Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"\n* Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):  # `source` captures node name
                _render_completed_message(update["messages"][-1])
