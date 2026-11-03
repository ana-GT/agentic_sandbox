from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.utils.uuid import uuid7
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

USER_DATABASE = {
  "user123": {
    "name" : "Ben Franklin",
    "account_type": "Premium",
    "balance": 5000,
    "email": "bfranklin@example.com"  
  },
  "user456": {
    "name" : "Martha Washington",
    "account_type": "Standard",
    "balance": 1200,
    "email": "mwash@example.com"      
  }
}

@dataclass
class UserContext:
  user_id: str
  
@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
  """Get the current user's account information"""
  user_id = runtime.context.user_id
  
  if user_id in USER_DATABASE:
    user = USER_DATABASE[user_id]
    return (
      f"Account holder: {user['name']}\n"
      f"Type: {user['account_type']}\n"
      f"Balance: ${user['balance']}" 
    )
  return "User not found"
  
@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Get the number of messages in the conversation."""
    messages = runtime.state["messages"]
    return f"There are {len(messages)} messages."  
  
model = ChatOllama(model="gemma4:e2b")
agent = create_agent(model, tools=[get_account_info, get_message_count], context_schema=UserContext,
        checkpointer=InMemorySaver(),
        system_prompt="You are a financial assistant."
)

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}
context = UserContext(user_id = "user123")

result = agent.invoke(
  {"messages": [{"role": "user", "content": "what is my current balance?"}]},
  config = config,
  context = context
)

print(result["messages"][-1].content_blocks)

result = agent.invoke(
  {"messages": [{"role": "user", "content": "what is my balance, again?"}]},
  config = config,
  context = context
)

print(result["messages"][-1].content_blocks)
        
result = agent.invoke(
  {"messages": [{"role": "user", "content": "How many messages have we exchanged, so far?"}]},
  config = config,
  context = context
)

print(result["messages"][-1].content_blocks)        
