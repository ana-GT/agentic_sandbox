# Updated from: https://www.geeksforgeeks.org/nlp/building-a-basic-pdf-summarizer-llm-application-with-langchain/

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog


def get_file_path():
    # Define the allowed file types (Optional)
    file_types = [
        ("PDF files", "*.pdf"),
    ]
    
    # Open the dialog and capture the full path
    full_path = filedialog.askopenfilename(
        title="Select a File",
        initialdir=".",
        filetypes=file_types
    )
    
    if full_path:
      return full_path
    else:
      return None

# ********************
# class Summarizer
# ********************
class Summarizer:

  def load(self, pdf_filepath : str) -> bool:
    loader = PyPDFLoader(pdf_filepath)
    self.docs = loader.load()
    
    if self.docs:
      return True
    else:
      return False
   
  def initialize(self, model: str="llama3.1:8b", temperature: float=0.3) -> bool:
    self.llm = OllamaLLM(model=model, temperature=temperature)
    return True
    
  def summarize(self) -> str:
    template = """
    You are an expert research assistant. Write a comprehensive, well-structured sumary of the following text. Your summary should include:
    - A high-level TL;DR paragraph
    - Key takeaways presented as a clean, bulleted list.
    - Critical insights or data points.
    
    Text to summarize: {text}
    
    SUMMARY:
    """
    
    prompt = PromptTemplate.from_template(template)
    full_text = "\n\n".join([doc.page_content for doc in self.docs])
    chain = prompt | self.llm | StrOutputParser()
    
    summary = chain.invoke({"text": full_text})
    return summary
    
########################################   
def main():

    st.set_page_config(page_title="Summarize a PDF", page_icon=None)
    
    st.title("Create a pdf summary")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pdf_filepath" not in st.session_state.keys():
      st.session_state["pdf_filepath"] = None

    if "model" not in st.session_state:
      st.session_state.model = None


    if "summarizer" not in st.session_state:
      st.session_state.summarizer = Summarizer()

    with st.sidebar:
        st.subheader("Upload Your Documents:")
                  
        # Select pdf          
        if st.button("Select PDF"):
          # Run the selector
          st.session_state.pdf_filepath = get_file_path()
          
        st.write(f"PDF filepath: {st.session_state.pdf_filepath}")
        
        # Model
        st.session_state.model = st.selectbox("Choose the model to use",
            options=["gemma4:e2b", "llama3.1:8b"]
        )
        
        # Load pdf
        if st.button("Initialize model"):
        
            if st.session_state.pdf_filepath is not None:
              st.session_state.summarizer.load(st.session_state.pdf_filepath)
            
              if st.session_state.summarizer.initialize(st.session_state.model):
                  st.success("Initialization complete! You can now get the summary of your PDFs.")
            else:
              st.error("Please click select pdf first")

    user_question = st.chat_input("Ask a question about your PDFs:")
    if user_question:
        response = st.session_state.summarizer.summarize()
        st.write(response)
        
if __name__ == '__main__':
    main()    
 
