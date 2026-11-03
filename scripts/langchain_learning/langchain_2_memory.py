import os

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def on_load(model : str="gemma4:e2b"):

  global chat, store
  
  chat = ChatOllama(
    model = model,
    temperature = 0
  )
  store = {}
  
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]

def test_buffer_mem():

  chain = RunnableWithMessageHistory(
    chat,
    get_session_history
  )
    
  session_id = "1"  
  response = chain.invoke("Hi my name is Ana", config={"configurable": {"session_id": session_id}})
  print(response.content)
  response = chain.invoke("What is 1+1?", config={"configurable": {"session_id": session_id}})
  print(response.content)
  response = chain.invoke("Do you remember my name?", config={"configurable": {"session_id": session_id}})
  print(response.content)
  
  # Check out what is stored
  for mi in store[session_id].messages:
    print(f"""Type: {mi.type}. Content: {mi.content}""")

from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
def test_buffer_window_mem():

  memory = ConversationBufferWindowMemory(k=1)
  langchain.agents.create_agent

# ***************
# Testing  
# ***************
on_load(model="gemma4:e2b")
#test_buffer_mem()
test_buffer_window_mem()
