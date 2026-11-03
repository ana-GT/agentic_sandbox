from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

from typing import Literal

class TestState(AgentState):
  user_name: str

@tool 
def set_user_name(new_name: str, runtime: ToolRuntime[None, TestState]) -> Command:
  """Set the user's name in the conversation state."""
  return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(
                    content=f"User name set to {new_name}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

 
config = {"configurable": {"thread_id": str(uuid7())}}
agent = create_agent(model="ollama:gemma4:e2b", tools=[set_user_name], state_schema=TestState, checkpointer=InMemorySaver())

messages = {
  "messages": [
    {"role": "system", "content": "How can I help you today?"},
    {"role": "user", "content": "Please set my user_name to olive_oil"}  
  ]
}

response = agent.invoke(messages, config=config)
print(response["messages"][-1].content)

messages = {
  "messages": [
    {"role": "user", "content": "Can you change my user name to Doggy Dog?"}  
  ]
}

response = agent.invoke(messages, config=config)
print(response["messages"][-1].content)


messages = {
  "messages": [
    {"role": "user", "content": "Can you tell me my user name?"}  
  ]
}

response = agent.invoke(messages, config=config)
print(response["messages"][-1].content)

