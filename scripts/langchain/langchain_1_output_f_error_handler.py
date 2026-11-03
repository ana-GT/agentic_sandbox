from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from typing import Union
from pydantic import BaseModel, Field
from langchain.messages import AIMessage, HumanMessage


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"


agent = create_agent(
    model="ollama:gemma4:e2b",
    tools=[],
    response_format=ToolStrategy(
                        schema=Union[ContactInfo, EventDetails],
                        handle_errors=custom_error_handler
                    )  # Default: handle_errors=True
)

#result = agent.invoke({
#    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
#})

#for msg in result['messages']:
#    # If message is actually a ToolMessage object (not a dict), check its class name
#    if type(msg).__name__ == "ToolMessage":
#        print(msg.content)
#    # If message is a dictionary or you want a fallback
#    elif isinstance(msg, dict) and msg.get('tool_call_id'):
#        print(msg['content'])
        
        
msgs = {"messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]}        
stream = agent.stream_events(msgs, 
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
      else:
        print(f"-- Messsage: {latest_message}") 
    elif latest_message.tool_calls:
      print(f"**-- Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")        
    else:
      print(f"-- Elif: {latest_message}")  
