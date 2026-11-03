from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

import requests

@tool
def get_weather(city: str) -> str:
  """Search online for the current weather in the city matching the input. Returns a 
  string with the weather information
  
  Args:
    city: The city for which we want the current weather
  """
  
  url = f"https://wttr.in/{city}?format=3"
  response = requests.get(url)
  
  if response.status_code == 200:
    return response.text
  else:
    return "Failed to retrieve weather data"

agent = create_agent(
  model = "ollama:gemma4:e2b",
  tools = [get_weather],
  checkpointer = InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
  {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]},
  config = config,
)


result = agent.invoke(
  {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
  config = config
)

result = agent.invoke(
  {"messages": [{"role": "user", "content": "What is the weather in Lima?"}]},
  config = config
)


print("CHECKING ***************")
i = 0
for mi in result["messages"]:
  print(f"* Message[{i}]: Type: {mi.type}. Content: {mi.content}")
  i = i + 1

print("CONTENT************")
print(result)
