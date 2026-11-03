from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

from typing import Literal

@tool 
def get_last_user_msg(runtime: ToolRuntime) -> str:
  """Get the most recent message from human user"""
  messages = runtime.state["messages"]
  
  for message in reversed(messages):
    if isinstance(message, HumanMessage):
      return message.content
      
  return "No human user messages found"
 

agent = create_agent(model="ollama:gemma4:e2b", tools=[get_last_user_msg])

messages = {
  "messages": [
    {"role": "system", "content": "How can I help you today?"},
    {"role": "user", "content": "Can you guess who will win the Grand Prix of Figure Skating this year in the women category?"}  
  ]
}

response = agent.invoke(messages)
print(response["messages"][-1].content)

messages = {
  "messages": [
    {"role": "user", "content": "Can you tell me the last message I wrote to you?"}  
  ]
}

response = agent.invoke(messages)
print(response["messages"][-1].content)

