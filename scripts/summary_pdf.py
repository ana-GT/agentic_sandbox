from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


class Summarizer:

  def load(self, pdf_filepath : str):
    loader = PyPDFLoader(pdf_filepath)
    self.docs = loader.load()
    print("Loaded pdf successfully")
   
  def initialize(self, model: str="llama3.1:8b", temperature: float=0.3):
    self.llm = OllamaLLM(model=model, temperature=temperature)
    print("Initialized Summarizer successfully")
    
  def summarize(self):
    template = """
    You are an expert research assistant. Write a comprehensive, well-structured sumary of the following text. Your summary should include:
    - A high-level TL;DR paragraph
    - Key takeaways presented as a clean, bulleted list.
    - Crhtical insights or data points.
    
    Text to summarize: {text}
    
    SUMMARY:
    """
    
    prompt = PromptTemplate.from_template(template)
    full_text = "\n\n".join([doc.page_content for doc in self.docs])
    chain = prompt | self.llm | StrOutputParser()
    
    summary = chain.invoke({"text": full_text})
    print(summary)
    
    
if __name__ == "__main__":

  pdf_filepath = "/home/ana/Research/agentic_sandbox/bibliography_bp_2026/integrators_paper_1.pdf"
  model = "gemma4:e2b" #"llama3.1:8b"
  
  summarizer = Summarizer()
  summarizer.load(pdf_filepath)
  summarizer.initialize(model)
  summarizer.summarize()
    
    


