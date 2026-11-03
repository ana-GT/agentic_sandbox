from langchain.agents import create_agent
from langchain_ollama import ChatOllama

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
         
model = "ollama:gemma4:e2b" # "ollama:devstral-2"    

chat = ChatOllama(model = model, temperature=0)

agent = create_agent(model=model, tools=[get_weather], system_prompt="You are a helpful assistant")

msg = {
  "messages": [ {"role": "user", "content": "What is the weather in SF?"} ]
}

result = agent.invoke(msg)
print(result["messages"][-1].content_blocks)
