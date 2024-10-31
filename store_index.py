from src.helper import extract_pdf,splitting,download_hugging_face_embeddings
from langchain_chroma import Chroma

document=extract_pdf("C:/GENAI/END-TO-END-MEDICAL-CHATBOT/data")
text_chunks=splitting(document)
embeddings=download_hugging_face_embeddings()

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="C:/GENAI/END-TO-END-MEDICAL-CHATBOT/vector_store",  # Where to save data locally, remove if not necessary
)

