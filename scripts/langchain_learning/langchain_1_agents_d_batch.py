import requests

from langchain_ollama import ChatOllama


# Global  
model = ChatOllama(model="gemma4:e2b", temperature=0)

def ask_questions(questions : list[str]) -> None:
  
  responses = model.batch(questions)
 
  for ri, qi in zip(responses, questions):
    print(f"****************Response to '{qi}' *********** ")
    print(ri)
    print(f"\n")
      
qs = ["Do parrots like to talk?", "How do you make marranito cookies?", "What is string theory?"]  
print(ask_questions(qs))
