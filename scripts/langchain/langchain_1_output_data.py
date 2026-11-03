from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
  """Contact information for a person"""
  name: str = Field(description="The name of the person")
  email: str = Field(description="The email address of the person")
  phone: str = Field(description="The pone number of the person")


agent = create_agent(
    model="ollama:gemma4:e2b",
    #tools=[get_user_info],
    #checkpointer=InMemorySaver(),
    response_format=ContactInfo
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]},
)
print(response["structured_response"])  # "You are Bob!"
print("***************")
print(response)
#response = agent.invoke(
#    {"messages": [{"role": "user", "content": "That is wrong, that is not my full name"}]},
#    thread_config,
#)["messages"][-1].content

#print(response)  # "You are Bob!"
