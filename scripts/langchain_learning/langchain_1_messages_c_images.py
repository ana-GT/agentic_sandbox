from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import base64

from langchain_core.load import dumpd, load

# Get image
def encode_image(image_path):
  with open(image_path, "rb",) as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

img_path = "/home/ana/Pictures/HaverHouse/dogs/marley/marley_2026_05_12.png"

model = init_chat_model("ollama:gemma4:e2b")

message = {
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe the content of this image"},
    {"type": "image",
     "base64": encode_image(img_path),
     "mime_type": "image/png"} 
  ]
}

#response = model.invoke([message])
#print(response.content)

message = HumanMessage("What is the capital of France?")

serialized = dumpd(message)
print("Serialized")
print(serialized)

restored = load(serialized)
print("Restored")
print(restored)
