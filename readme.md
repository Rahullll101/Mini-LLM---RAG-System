<!-- def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks using recursive, structure-aware chunking.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show one example chunk (for demo/debug)
    if split_docs:
        print("\nExample chunk:")
        print(split_docs[0].page_content[:200], "...")
        print("Metadata:", split_docs[0].metadata)

    return split_docs -->