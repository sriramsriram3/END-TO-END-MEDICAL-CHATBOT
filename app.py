from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings,llm_model
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv
from store_index import initialize_chroma_vector_store
from langchain.prompts import PromptTemplate

load_dotenv()
api_key=os.getenv('GROQ_API_KEY')
embeddings=download_hugging_face_embeddings()
llm=llm_model(api_key)
vector_store=initialize_chroma_vector_store()
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

input='what are allergies'
result=qa({"query": input})
print(result)