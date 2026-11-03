

# Updated from: https://www.geeksforgeeks.org/nlp/building-a-basic-pdf-summarizer-llm-application-with-langchain/

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      extracted_text = page.extract_text()
      if extracted_text:
        text += extracted_text + "\n"
        
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)
    
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
if __name__ == "__main__":
  print("Testing...")
  
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        huggingfacehub_api_token=HF_TOKEN
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )  
    
def handle_userinput(user_question):
    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.error("Please upload and process a PDF first.")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.subheader("Chat History")
    for i, message in enumerate(st.session_state.chat_history):
        speaker = "You:" if i % 2 == 0 else "AI:"
        st.write(f"**{speaker}** {message.content}")    
    
def main():
    st.set_page_config(page_title="Brainstorm with PDFs", page_icon=None)
    st.title("Brainstorm with Your PDFs")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete! You can now chat with your PDFs.")

    user_question = st.chat_input("Ask a question about your PDFs:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()    
    
