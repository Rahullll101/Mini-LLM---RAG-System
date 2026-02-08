# main.py
import streamlit as st

from Loader import load_pdfs
from chunk import semantic_split_documents
from embeding_db import load_embedding_model, create_vector_db
from llm import load_llm
from Rag_app import answer_query


@st.cache_resource
def initialize_rag():
    documents = load_pdfs("data/pdf")

    semantic_chunks = semantic_split_documents(
        documents,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = load_embedding_model()
    vector_db = create_vector_db(semantic_chunks, embeddings)
    llm_instance = load_llm()

    return vector_db, llm_instance


vector_db, llm = initialize_rag()

st.title(" RAG Document Assistant")
query = st.text_input("Ask a question")

if query:
    answer, docs = answer_query(query, vector_db, llm)
    st.write(answer.content)

    st.subheader("Sources")
    for doc in docs:
        st.write(
            f"- {doc.metadata.get('source')}"
        )
