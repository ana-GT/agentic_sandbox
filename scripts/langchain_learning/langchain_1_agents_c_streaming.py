from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.messages import AIMessage, HumanMessage
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

@dataclass
class Context:
  user_id: str

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
  user_id = request.runtime.context.user_id
  return f"You are an agent that address the user by its user id (which is {user_id}). Use a playful language"


# Global  
agent = create_agent(
  model="ollama:gemma4:e2b",
  tools=[get_weather],
  context_schema=Context,
  checkpointer=InMemorySaver(),
  middleware=[personalized_prompt]
)
config = {"configurable": {"thread_id": str(uuid7())}}

def ask_question(question : str):
  
   
  msgs = {"messages": [{"role": "user", "content": question}]}
  
  stream = agent.stream_events(msgs, 
    config = config,
    context = Context(user_id="ana_h"),
    version="v3"
  )

  for snapshot in stream.values:
    latest_message = snapshot["messages"][-1]
    if latest_message.content:
      if isinstance(latest_message, HumanMessage):
        print(f"**-- User: {latest_message.content}")
      elif isinstance(latest_message, AIMessage):
        if latest_message.content[-1]['type'] == 'text':
          print(f"**-- Agent: {latest_message.content[-1]['text']}")
        elif latest_message.content[-1]['type'] == 'tool_call':
          print(f"**-- Tool: {latest_message.content[-1]['name']}")  
    elif latest_message.tool_calls:
      print(f"**-- Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

  
ask_question("What is the weather in San Francisco?")
