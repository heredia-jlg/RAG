from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import os
from utils import split_documents
from EmbeddingManager import EmbeddingManager
from VectorStore import VectorStore

document = Document(page_content="Hello, World!",
                    metadata={"source": "test.txt"})

pdf_loader = DirectoryLoader(
    path="./PDFs",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)
pdf_documents = pdf_loader.load()

chunks =split_documents(pdf_documents)


#embedding_manager = EmbeddingManager()
#vector_store = VectorStore()

#if __name__ == "__main__":
#    print("Hello, World!")
