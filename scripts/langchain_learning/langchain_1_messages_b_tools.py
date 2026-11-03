from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

model = init_chat_model("ollama:gemma4:e2b")

ai_message = AIMessage(
  content=[],
  tool_calls = [
    {"name": "get_weather",
     "args": {"location": "San Francisco"},
     "id": "call_123"
    }  
  ]
)

weather_result = "Sunny, 72F"
tool_message = ToolMessage(
  content = weather_result,
  tool_call_id = "call_123"
)

# Conversation
messages = [
  HumanMessage("What is the weather in San Francisco?"),
  ai_message,
  tool_message
]

#response = model.invoke(messages)
#print(response.content)

from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"},
            ],
        },
        {"type": "text", "text": "...", "id": "msg_abc123"},
    ],
    response_metadata={"model_provider": "openai"}
)
print(message.content_blocks)


