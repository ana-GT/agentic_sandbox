from ollama import chat, ChatResponse

def get_temperature(city: str) -> str:
  """Get the current temperature for a city
  
  Args:
    city: The name of the city

  Returns:
    The current temperature for the city
  """
  temperatures = {
    "New York": "22°C",
    "London": "15°C",
    "Tokyo": "18°C",
  }
  return temperatures.get(city, "Unknown")

def get_conditions(city: str) -> str:
  """Get the current weather conditions for a city
  
  Args:
    city: The name of the city

  Returns:
    The current weather conditions for the city
  """
  conditions = {
    "New York": "Partly cloudy",
    "London": "Rainy",
    "Tokyo": "Sunny"
  }
  return conditions.get(city, "Unknown")


def add(a: int, b: int) -> int:
  """Add two numbers"""
  """
  Args:
    a: The first number
    b: The second number

  Returns:
    The sum of the two numbers
  """
  return a + b


def multiply(a: int, b: int) -> int:
  """Multiply two numbers"""
  """
  Args:
    a: The first number
    b: The second number

  Returns:
    The product of the two numbers
  """
  return a * b


def test_chatbot2(operation: str, model: str="gemma4:e2b"):

  available_functions = {
    'add': add,
    'multiply': multiply,
  }

  user_prompt = f"""
  What is {operation}?
  """
  
  messages = [{'role': 'user', 
               'content': user_prompt}]

  while True:
    response: ChatResponse = chat(
        model=model,
        messages=messages,
        tools=[add, multiply],
        think=True,
    )
    messages.append(response.message)
    print("-- Thinking: ", response.message.thinking)
    print("Content: ", response.message.content)
    if response.message.tool_calls:
        for tc in response.message.tool_calls:
            if tc.function.name in available_functions:
                print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
                result = available_functions[tc.function.name](**tc.function.arguments)
                print(f"Result: {result}")
                # add the tool result to the messages
                
                messages.append({'role': 'tool', 
                		 'tool_name': tc.function.name, 
                		 'content': str(result)})
    else:
        # end the loop when there are no more tool calls
        break
  # continue the loop with the updated messages


################################
def test_chatbot(city1: str, city2: str, model: str="gemma4:e2b") -> str:

  user_prompt = f"""
  "What are the current weather conditions and  temperature in {city1} and {city2}?"
  """

  messages = [{"role": "user", "content": user_prompt}]

  # pass functions directly as tools in the tools list or as a JSON schema
  response = chat(model=model, 
  	messages=messages, 
  	tools=[get_temperature, get_conditions], 
  	think=True)

  messages.append(response.message)
  
  
  if response.message.tool_calls:
    # process each tool call
    for call in response.message.tool_calls:
      if call.function.name == "get_temperature":
        result = get_temperature(**call.function.arguments)
      elif call.function.name == "get_conditions":
        result = get_conditions(**call.function.arguments)
      else:
        result = 'Unknown tool'
        
      # add the tool result to the messages
      messages.append({"role": "tool", 
                       "tool_name": call.function.name, 
                       "content": str(result)})

  final_response = chat(model=model, 
    	messages=messages, 
    	tools=[get_temperature, get_conditions], think=True)
    	
  print(final_response.message.content)   

#############################
def test_chatbot3(city1: str, city2: str, model: str="gemma4:e2b"):
  available_functions = {
    'get_temperature': get_temperature,
  }
  
  user_prompt = f"""
  "What are the temperatures in {city1} and {city2}?"
  """

  messages = [{"role": "user", "content": user_prompt}]

  
  # directly pass the function as part of the tools list
  response = chat(model=model, messages=messages, tools=available_functions.values(), think=True)
  print(response.message)
  
# Actually call the function
#test_chatbot("London", "New York", "gemma4:e2b")
#test_chatbot2("(11434+12341)*412", "gemma4:e2b")
test_chatbot3("London", "New York", "gemma4:e2b")
