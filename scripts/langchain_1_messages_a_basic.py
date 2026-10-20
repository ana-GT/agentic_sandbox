from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("ollama:gemma4:e2b")

system_msgs = SystemMessage("You are a helpful assistant")
human_msgs = HumanMessage("Hi, what is your name and how old are ya?")

msgs = [system_msgs, human_msgs]
#response = model.invoke(msgs)
#print(response.content)

msgs_2 = [
  SystemMessage("You are a poetry expert, Gemma"),
  HumanMessage("Write a haiku about spring"),
  AIMessage("Cherry blossoms")
]
#response = model.invoke(msgs_2)
#print(response.content)

# Use dictionary format
msgs_3 = [
  {"role": "system", "content": "You are a poetry expoert"},
  {"role": "user", "content": "Write a haiku about spring"},
  {"role": "assistant", "content": "Cherry blossoms bloom..."}
]

#response = model.invoke(msgs_3)
#print(response.content)

msgs_4 = [
  SystemMessage("You are a helpful coding assistant"),
  HumanMessage("How do I create a REST API?")
]
#response = model.invoke(msgs_4)
#print(response.content)

#response = model.invoke("Explain AI in a paragraph")
#print(response)


ai_msg = AIMessage("I'd be happy to help you with that question")
msgs_5 = [
   SystemMessage("You are a helpful assistant"),
   HumanMessage("Can you help me?"),
   ai_msg,
   HumanMessage("Great, what is 2+59?")
]

#response = model.invoke(msgs_5)
#print(response.content)

# Tool calls
def get_weather(location: str) -> str:
 return f"""Weather in {location} is sunny today"""
 
model_with_tools = model.bind_tools([get_weather])
#response = model_with_tools.invoke("What is the weather in Paris?")

#for tool_call in response.tool_calls:
#  print(f"Tool: {tool_call['name']}")
#  print(f"Args: {tool_call['args']}")
#  print(f"ID: {tool_call['id']}")

#print(f"Token usage to create this response: {response.usage_metadata}")

chunks = []
full_message = None

for chunk in model.stream("Hi"):
  if chunk.text:
    chunks.append(chunk)
    print("Chunk: " + chunk.text)
  full_message = chunk if full_message is None else full_message + chunk
  
print(full_message.content)  
