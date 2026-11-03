from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool

@tool("calculator", description="Performsarithmetic calculations. Use this for math problems" )
def calc(query: str) -> str:
  """Evaluate arithmetic operations"""
  res = eval(query)
  res_str = str(res)
  print(f"The answer is: {res_str}")
  return res_str
 
model = init_chat_model("ollama:gemma4:e2b")
model_with_tools = model.bind_tools([calc])

response_1 = model_with_tools.invoke("How much is '45*66'?")
if response_1.tool_calls:
  fx_output = calc.invoke(response_1.tool_calls[0]["args"]["query"])
  print(fx_output)
