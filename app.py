from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings,llm_model
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv
from store_index import initialize_astradb_vector_store
from langchain.prompts import PromptTemplate

load_dotenv()
api_key=os.getenv('GROQ_API_KEY')
embeddings=download_hugging_face_embeddings()
llm=llm_model(api_key,temperature=0.25)
vector_store=initialize_astradb_vector_store()
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

# Create the Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)