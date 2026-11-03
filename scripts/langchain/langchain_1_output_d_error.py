from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="ollama:gemma4:e2b",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails], handle_errors=False)  # Default: handle_errors=True
)

print("Invoking response")
response = agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})
print("End of invoking response")

print("Pringint response...")
print(response["structured_response"])

