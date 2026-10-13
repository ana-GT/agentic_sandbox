import os
from openai import OpenAI
import datetime

from dotenv  import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.structured import ResponseSchema
from langchain_classic.output_parsers import StructuredOutputParser

# If we were using OpenAI, we'd use this to load our key
# we are using Llama and local models, so no need for it
def on_load(model : str="gemma4:e2b"):

  _ = load_dotenv(find_dotenv())
  current_date = datetime.datetime.now().date()

  global client
  client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
  )
  
  global chat
  chat = ChatOllama(
    model = model, 
    temperature=0
  )

  template_string = """
  Translate the text \
  that is delimited by triple backticks \
  into a style that is {style}. \
  text: ``` {text} ```
  """
  global prompt_template
  prompt_template = ChatPromptTemplate.from_template(template_string)


def get_completion(text: str, style: str, model : str="gemma4:e2b"):
  messages = [{"role": "user", 
               "content": prompt}]

  prompt = f"""
  Translate the text that is delimited \
  by triple backticks into a style that is \
  {style}, \
  text:  \
  ```{customer_input}```
  """


  global client
  response = client.chat.completions.create(
    model = model,
    messages = messages,
    temperature = 0)
  return response.choices[0].message.content

# *********************
# Format input
# **********************
def get_answer_langchain(input_text : str, output_style : str):
  
  global prompt_template
  formatted_messages = prompt_template.format_messages(style = output_style, text = input_text)
  
  print("Input received*****")
  print(formatted_messages)

  # Call the LLM to translate
  response = chat.invoke(formatted_messages) 
  print("Response****") 
  print(response.content)

# *********************
# Output parsers
# **********************
def output_parser():
  input_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
  """
  
  output_template = """\
  For the following text, extract the following information:

  gift: Was the item purchased as a gift for someone else? \
  Answer True if yes, False if not or unknown.

  delivery_days: How many days did it take for the product \
  to arrive? If this information is not found, output -1.

  price_value: Extract any sentences about the value or price,\
  and output them as a comma separated Python list.

  Format the output as JSON with the following keys:
  gift
  delivery_days
  price_value

  text: {text}
  """
  
  #prompt_template = ChatPromptTemplate.from_template(output_template)
  #print(prompt_template)
  
  #messages = prompt_template.format_messages(text=input_review)
  #print("Messages ****")
  #print(messages)
  #response = chat.invoke(messages)
  #print("Response content: ********")
  #print(response.content)
  #print("Type of response content **********")
  #print(type(response.content))
  
  gift_schema = ResponseSchema(name="gift", 
  		description="Was the item purchased for someone else? \
  		Answer True if yes, \
  		False if not or unknown")
  delivery_schema = ResponseSchema(name="delivery_days",
  		description="How many days\
  		did it take for the product\
  		to arrive? If this information is not found, output -1")
  price_value_schema = ResponseSchema(name="price_value",
  		description="Extract any\
  		sentences about the value or\
  		price , and output them as a \
  		comma separated Python list")
  
  response_schemas = [gift_schema, delivery_schema, price_value_schema]
  
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()
  print("Format instructions **********")
  print(format_instructions)

  output_template_2 = """\
  For the following text, extract the following information:

  gift: Was the item purchased as a gift for someone else? \
  Answer True if yes, False if not or unknown.

  delivery_days: How many days did it take for the product \
  to arrive? If this information is not found, output -1.

  price_value: Extract any sentences about the value or price,\
  and output them as a comma separated Python list.

  text: 
  {text}
  
  format instructions:
  {format_instructions}
  """
  
  prompt_template_2 = ChatPromptTemplate.from_template(template=output_template_2)

  messages = prompt_template_2.format_messages(text=input_review, 
                                format_instructions=format_instructions)

  print("Message content *****")
  print(messages[0].content)	
  response_2 = chat.invoke(messages)
  print("response:")
  print(response_2.content)
  output_dict = output_parser.parse(response_2.content)
  
  print("Output dict")
  print(output_dict)
  print("Type output dict")  
  print(type(output_dict))
  print("Get delivery days")
  print(output_dict.get('delivery_days'))  
# ****************    
# Test
# ****************
input_text_1 = """
O Caramelo e um amigo, ele nao e un perigo, so corre tras dos motocas que barulhan seus ouvidos. Brasileiro! E un vira-lata que se diz, respeita o Caramelo que tenha a nossa raiz. Ele da un pulinho e abana o rabo, que cachorrinho engrasado
"""
output_style_1 = """
English in a calm and respectful tone
"""

input_text_2 = """
Patinaje sobre hielo es un deporte apasionante pero no muy extendido en la poblacion \
es posible que su limitada disponibilidad en los medios de comunicacion, combinado \
con los costos que no permiten que sea un deporte muy practicado entre ninos lo que contribuya a que sea considerado un deporte algo exclusivo. Aun asi, me gusta mucho.
"""

output_style_2 = """
Italian in a youthful tone
"""



on_load("gemma4:e2b")
#response = get_completion(input_text, output_style, prompt, "gemma4:e2b")
#print(response)

#print("********************")
#print("Get answer langchain 1----")
#get_answer_langchain(input_text_1, output_style_1)
#print("Get answer langchain 2-----")
#get_answer_langchain(input_text_2, output_style_2)

print("********************")
output_parser()
