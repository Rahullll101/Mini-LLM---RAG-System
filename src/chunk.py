#data chunking 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter #sematic



# # chunk.py
# from langchain_text_splitters import (
#     RecursiveCharacterTextSplitter,
#     SentenceTransformersTokenTextSplitter
# )
# from langchain.schema import Document


# def split_documents(
#     documents,
#     chunk_size: int = 1000,
#     chunk_overlap: int = 200
# ):
#     """
#     Recursive (structure-aware) chunking.
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return splitter.split_documents(documents)


#-------------------------------------------



def semantic_split_documents(
    documents,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500
):
    """
    Semantic chunking using sentence-transformer embeddings.
    """
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=0
    )

    semantic_chunks = []

    for doc in documents:
        for chunk in splitter.split_text(doc.page_content):
            semantic_chunks.append(
                doc.__class__(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    return semantic_chunks




































# def semantic_split_documents(
#     documents,
#     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#     chunk_size=500
# ):
#     """
#     Semantic chunking using sentence-transformer embeddings.
#     Groups semantically similar sentences together.
#     """

#     splitter = SentenceTransformersTokenTextSplitter(
#         model_name=embedding_model,
#         chunk_size=chunk_size,
#         chunk_overlap=0
#     )

#     semantic_docs = []

#     for doc in documents:
#         chunks = splitter.split_text(doc.page_content)

#         for chunk in chunks:
#             semantic_docs.append(
#                 doc.__class__(
#                     page_content=chunk,
#                     metadata=doc.metadata
#                 )
#             )

#     print(f"Semantic split of {len(documents)} documents into {len(semantic_docs)} chunks")

#     # Example chunk
#     if semantic_docs:
#         print("\nExample semantic chunk:")
#         print(semantic_docs[0].page_content[:200], "...")
#         print("Metadata:", semantic_docs[0].metadata)

#     return semantic_docs
# semantic_chunks = semantic_split_documents(pdf_doc)
# # semantic_chunks




# def split_documents(documents, chunk_size=1000, chunk_overlap=200):
#     """
#     Split documents into smaller chunks using recursive, structure-aware chunking.
#     """

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )

#     split_docs = text_splitter.split_documents(documents)

#     print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

#     # Show one example chunk (for demo/debug)
#     if split_docs:
#         print("\nExample chunk:")
#         print(split_docs[0].page_content[:200], "...")
#         print("Metadata:", split_docs[0].metadata)

#     return split_docs

# chunks=split_documents(pdf_doc)
# chunks