from src.helper import extract_pdf,splitting,download_hugging_face_embeddings
from langchain_chroma import Chroma
from uuid import uuid4

document=extract_pdf("C:/GENAI/END-TO-END-MEDICAL-CHATBOT/data")
text_chunks=splitting(document)
embeddings=download_hugging_face_embeddings()

def initialize_astradb_vector_store():
    """
    Initialize the Chroma (AstraDB) vector store with the specified collection and persistence directory.
    
    Parameters:
    embeddings: The embeddings object used to create vector representations of data.
    """
    # Import AstraDBVectorStore
    from langchain_astradb import AstraDBVectorStore
    import os
    
    # Retrieve AstraDB credentials from environment variables
    astra_token = os.getenv('ASTRADB_TOKEN')
    astradb_endpoint = os.getenv('ASTRADB_API_ENDPOINT')
    
    # Initialize the vector store with the specified parameters
    vector_store = AstraDBVectorStore(
        collection_name="astra_vector_langchain",
        embedding=embeddings,
        api_endpoint=astradb_endpoint,
        token=astra_token,
    )
    
    return vector_store


vector_store=initialize_astradb_vector_store()
uuids = [str(uuid4()) for _ in range(len(text_chunks))]
vector_store.add_documents(documents=text_chunks, ids=uuids)