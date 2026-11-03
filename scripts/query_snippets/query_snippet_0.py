"""
# My first app
Here's our first attempt at using data to create a table
"""
import streamlit as st
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver


class QueryAnswer :

  def __init__(self):
    self.agent = None
    self.config = None
 
  def initialize(self, model: str="ollama:gemma4:e2b"):

    system_prompt = """
    You are a helpful assistant that answers exactly what is asked. Do not add further instructions if you are not specifically asked to do so. Only reply with what you are asked.
    """

    self.agent = create_agent(
      model = model,
      #tools = [get_weather],
      checkpointer = InMemorySaver(),
      system_prompt=system_prompt
    )
    self.config = {"configurable": {"thread_id": str(uuid7())}}


  def handle_user_query(self, user_query: str) -> str:

    self.initialize()
    # Ask question
    response = self.agent.invoke(
    {"messages": [{"role": "user", "content": user_query}]},
    config = self.config,
    )

    return response["messages"][-1].content

# ************************
# main function
# ************************
def main():

    qa = QueryAnswer()
    
    st.set_page_config(page_title="One-shot Snippet queries", page_icon=None)
    st.title("Ask simple, one-shot queries:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    messages = st.container(height=500)
    user_query = st.chat_input("Write a query:")
    if user_query:
        messages.chat_message("user").write(user_query)
        response = qa.handle_user_query(user_query)
        messages.chat_message("assistant").write(response)


if __name__ == '__main__':
    main()    
    
