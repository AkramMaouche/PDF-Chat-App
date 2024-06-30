import streamlit as st  
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os  

from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PipelinePromptTemplate 
from dotenv import load_dotenv 

load_dotenv() 

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs: 
        pdfreader = PdfReader(pdf) ## Read multiple pages from the importing pdf  
        for page in pdfreader.pages: 
            text+=page.extract_text() ## extract all the information from  pages 
    return text 

def get_text_chunks(text): 
    text_splitter = RecursiveCharacterTextSplitter(chuk_size =10000,chunk_overlap=1000) 
    chunks = text_splitter.split_text(text)
    return chunks 

def getvectors_stors(text_chunks): ## transform our chunks to vectors
    embadding = GoogleGenerativeAIEmbeddings(models = "models/embedding-001") 
    vector_store = FAISS.from_texts(text_chunks,embaddings= embadding)
    vector_store.save_local("faiss.index")

def get_conversational_chain(): 
    prompt_template = ''' 
    Answer the question as detailed as possible from the provided context
    Context:\n {context}?\n 
    Question: \n {question}\n

    Answer :   

    '''





