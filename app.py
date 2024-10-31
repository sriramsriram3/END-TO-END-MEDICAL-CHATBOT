from flask import Flask,render_template,jsonify
from src.helper import download_hugging_face_embeddings,llm_model
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv
from store_index import initialize_chroma_vector_store

load_dotenv()
api_key=os.getenv('GROQ_API_KEY')
app=Flask(__name__)
embeddings=download_hugging_face_embeddings()
llm=llm_model(api_key)
vector_store=initialize_chroma_vector_store()
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])


