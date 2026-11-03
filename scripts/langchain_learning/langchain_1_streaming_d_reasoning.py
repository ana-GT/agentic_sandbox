from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.runnables import Runnable


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


model = ChatOllama(
    model="gemma4:e2b",
    timeout=None,
    stop=None,
    reasoning=True,
    thinking={"type": "enabled", "budget_tokens": 5000},
)
agent: Runnable = create_agent(
    model=model,
    tools=[get_weather],
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    version="v3",
)

for message in stream.messages:
    print("\n**** Reasoning: ", end="\n")
    for token in message.reasoning:
        print(f"{token}", end="")
    print("\n**** Message: ", end="\n")
    for token in message.text:
        print(token, end="", flush=True)
    print("\n")
