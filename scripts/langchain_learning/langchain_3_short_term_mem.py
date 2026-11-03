from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, SummarizationMiddleware, dynamic_prompt, ModelRequest
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any, TypedDict

from pydantic import BaseModel
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage

def global_load(model: str = "gemma4:e2b"):
  global local_llm, config
  local_llm = ChatOllama(model=model, temperature=0)
  config = {"configurable": {"thread_id": "1"}}

def get_user_info() -> str:
  """ Look up information about the current user"""
  return "No userprofile on file"
  
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
  """ Keep only the last few messages """ 
  messages = state["messages"]
  
  if len(messages) <= 3:
    return None 
    
  first_msg = messages[0]
  recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
  new_messages = [first_msg] + recent_messages
    
  return {
    "messages": [
      RemoveMessage(id=REMOVE_ALL_MESSAGES),
      *new_messages
    ]
  }
  
def test_agent(input_prompt : str):

  global local_llm
  agent = create_agent(model=local_llm, tools = [get_user_info], checkpointer=InMemorySaver(),
  )
  
  config_id = "1"
  thread_config = {"configurable": {"thread_id": config_id}}
  response = agent.invoke(
    {"messages": [{"role": "user", "content": input_prompt}]},
    thread_config,
    )
    
  # Print message replying to input
  print(response["messages"][-1].content)
  
  response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    thread_config
  )
  
  # Print second response
  print(response["messages"][-1].content)
  
def test_trim_agent():
  global local_llm, config  
  agent = create_agent(model=local_llm, tools = [], middleware=[trim_messages], checkpointer=InMemorySaver() )
  
  
  rp = agent.invoke({"messages": "Hi my name is Moonchild"}, config)
  agent.invoke({"messages": "Write a short poem about Houston"}, config)
  agent.invoke({"messages": "Now do the same but about Lima"}, config)
  
  final_response = agent.invoke({"messages": "What is my name?"}, config)
  
  print("Go over final response message")
  #for mi in final_response["messages"]:    
  #	mi.pretty_print()
  final_response["messages"][-1].pretty_print()

def stream(user_input : str, agent):

  global config
  stream = agent.stream_events(
    {"messages": [{"role": "user", "content": user_input}]},
    config, version="v3"
  )    
  for snapshot in stream.values:
    print([(message.type, message.content) for message in snapshot["messages"]])

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

  
def test_forgetful_agent():
  global local_llm  
  agent = create_agent(model=local_llm, tools = [], 
  	middleware=[delete_old_messages], checkpointer=InMemorySaver() )
  
  stream("hi, I am Estrella", agent)
  stream("write a short poem about pups", agent)
  stream("what is my name", agent)
  
def test_summary_agent():
  global local_llm, config
  agent = create_agent(model=local_llm, tools=[],
  	middleware=[
  	SummarizationMiddleware(
  	  model=local_llm,
  	  summary_prompt="sUMMARIZE TEXT",
  	  trigger=("tokens", 4000),
  	  keep=("messages", 20)
  	)
  	],
  	checkpointer=InMemorySaver()
 	)  
  
  response = agent.invoke({"messages": "Hi my name is Bill"}, config)
  print("Response 1:")
  for mi in response["messages"]:
    mi.pretty_print()

  response = agent.invoke({"messages": "What is the origin of the name Adam?"}, config)
  print("Response 2:")
  for mi in response["messages"]:
    mi.pretty_print()
  
  response = agent.invoke({"messages": "Now do the same but for the name Pratyush"}, config)
  print("Response 3:")
  for mi in response["messages"]:
    mi.pretty_print()

    
  final_response = agent.invoke({"messages": "What is my name"}, config)
  
  final_response["messages"][-1].pretty_print()

# *********************************
class CustomState(AgentState):
  user_name: str
  
class CustomContext(BaseModel):
  user_id: str

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState], ) -> Command:
  """Look up and update user info"""
  user_id = runtime.context.user_id
  name = "John Smith" if user_id == "user_123" else "Unknown user"
  return Command(update={
  	"user_name": name,
  	# update message history
  	"messages": [
  	  ToolMessage(
  	    "Successfully looked up user information",
  	    tool_call_id=runtime.tool_call_id
  	  )
  	]
  })
  
@tool
def greet(runtime: ToolRuntime[CustomContext, CustomState]) -> str | Command:
  """Use this to greet the user once you found their info"""  
  user_name = runtime.state.get("user_name", None)
  if user_name is None:
    return Command(update={
      "messages": [
        ToolMessage("Please call the 'update_user_info' tool, it will get an update the user name",
        tool_call_id = runtime.tool_call_id)
      ]
    })
    
  return f"Hi there {user_name}!!!"
  
def test_short_tools():
  global local_llm, config
  agent = create_agent(model=local_llm, 
  	tools=[update_user_info, greet],
  	state_schema=CustomState,
  	context_schema=CustomContext,)

  response = agent.invoke(
  	{"messages": [{"role": "user", "content" : "Greet the user!"}]},
  	context=CustomContext(user_id="user_123"),
  )
  
  print("Response message?")
  response["messages"][-1].pretty_print()

  response = agent.invoke(
  	{"messages": [{"role": "user", "content" : "Greet the user!"}]},
  	context=CustomContext(user_id="user_235"),
  )
  
  print("Response message?")
  response["messages"][-1].pretty_print()

##########################
#

class CustomContext2(TypedDict):
  user_name: str
  
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"  

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

def test_dyn_prompt():

  global local_llm, config
  agent = create_agent(model=local_llm, 
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext2,
  )

  result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext2(user_name="Madison Lyons"),
  )
  for msg in result["messages"]:
    msg.pretty_print()

  
model="gemma4:e2b"
global_load(model)
#test_agent("Hi, my name is Skeletor")
#test_trim_agent()
#test_forgetful_agent()
#test_summary_agent()
#test_short_tools()
test_dyn_prompt()
