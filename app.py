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

# Route for the home page with form
@app.route('/')
def index():
    return render_template('templates/chat.html')

# Route to handle the query and get the response
@app.route('/ask', methods=['POST'])
def ask():
    input_query = request.form.get('query')
    result = qa.invoke({"query": input_query})
    response = result["result"]
    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)