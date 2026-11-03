from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

def get_user_info() -> str:
    """Look up information about the current user."""
    return "Id is u123" #"No user profile on file."

agent = create_agent(
    "ollama:gemma4:e2b",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})


result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is my user id?"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})
print(result["messages"][-1].content)

