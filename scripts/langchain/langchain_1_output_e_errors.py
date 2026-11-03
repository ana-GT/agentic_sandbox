from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

agent = create_agent(
    model="ollama:gemma4:e2b",
    tools=[],
    response_format=ToolStrategy(ProductRating, handle_errors=False),  # Default: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do make not any field or value up."
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/100!"}]
#    "messages": [{"role": "user", "content": "Parse this: Amazing product, 4 out of 5 stars!"}]    
})


print(response["messages"][-1].text)
