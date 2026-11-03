from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool

from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
  """Input for weather queries"""
  location: str = Field(description="City name or coordinates")
  units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="Temperature unit preference")
  include_forecast: bool = Field(
    default=False,
    description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool=False) -> str:
  temp = 22 if units == "celsius" else 72
  result = f"Current weather in {location} is: {temp} degrees {units[0].upper()}"
  if include_forecast:
    result += ". Next 6 days: Stormy weather"
  return result
 
model = init_chat_model("ollama:gemma4:e2b")
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What is the current weather in Lima, and what is the forecast for the short term?")
if response.tool_calls:
  fx_output = get_weather.invoke(response.tool_calls[0]["args"])
  print(fx_output)
