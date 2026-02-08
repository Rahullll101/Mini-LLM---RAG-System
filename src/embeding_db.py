# embeding_db.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Load HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_db(
    chunks,
    embeddings,
    persist_directory: str = "/db/chroma_db"
):
    """
    Create a Chroma vector database from chunks.
    """
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectordb
