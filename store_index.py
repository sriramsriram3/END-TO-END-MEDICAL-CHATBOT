from src.helper import extract_pdf,splitting,download_hugging_face_embeddings
from langchain_chroma import Chroma
from uuid import uuid4

document=extract_pdf("C:/GENAI/END-TO-END-MEDICAL-CHATBOT/data")
text_chunks=splitting(document)
embeddings=download_hugging_face_embeddings()


def initialize_chroma_vector_store():
    # Initialize the Chroma vector store with the specified collection and persistence directory
    vector_store = Chroma(
        collection_name='example_collection',
        embedding_function=embeddings,
        persist_directory='C:/GENAI/END-TO-END-MEDICAL-CHATBOT/vector_store'
    )
    return vector_store


vector_store=initialize_chroma_vector_store()
uuids = [str(uuid4()) for _ in range(len(text_chunks))]
vector_store.add_documents(documents=text_chunks, ids=uuids)