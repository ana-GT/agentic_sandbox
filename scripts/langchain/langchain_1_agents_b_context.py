from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
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

def ask_question(question : str) -> str:
  
   
  msgs = {"messages": [{"role": "user", "content": question}]}
  result = agent.invoke(msgs, 
    config = config,
    context = Context(user_id="ana_h")
  )

  return result["messages"][-1].content
  
print(ask_question("What is the weather in San Francisco?"))
