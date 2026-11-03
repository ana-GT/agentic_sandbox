import os
from anthropic import Anthropic


client = Anthropic(
base_url="http://localhost:11434",
api_key="ollama"
)

# Create a conversation completion using a local model
message = client.messages.create(
    model="gemma4:e2b",  # Specify your local Ollama model name here
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Write a python function to check if a number is prime.",
        }
    ],
)

for block in message.content:
  if block.type == "text":
    print(block.text)

