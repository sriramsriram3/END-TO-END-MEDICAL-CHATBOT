# DATA INGESTION
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader  

def extract_pdf(pdf):
    loader=DirectoryLoader(pdf,glob='*pdf',loader_cls=PyPDFLoader)

    document=loader.load()
    return document


#splitting the data into chunks to ensure model supporting tokens
from langchain.text_splitter import RecursiveCharacterTextSplitter
def splitting(document):
    split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    splitted_doc=split.split_documents(document)
    return splitted_doc

from langchain_huggingface import HuggingFaceEmbeddings
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# from your_library import ChatGroq  # Import ChatGroq from the library where it is implemented

def llm_model():
    api_key = os.getenv("GROQ_API_KEYY")  # Securely retrieve API key from environment
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        groq_api_key=api_key
    )
    return llm

