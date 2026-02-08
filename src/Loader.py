#data loaders
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader



## load all the pdf files from the directory
def load_pdfs(pdf_dir: str):
    """
    Load all PDF documents from a directory.
    Returns a list of LangChain Document objects.
    """
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    return documents

# print(len(pdf_doc))
# print(pdf_doc)