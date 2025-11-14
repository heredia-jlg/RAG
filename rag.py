from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import os
from EmbeddingManager import EmbeddingManager

document = Document(page_content="Hello, World!",
                    metadata={"source": "test.txt"})

odf_loader = DirectoryLoader(
    path="./PDFs",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)
pdf_documents = odf_loader.load()

print("done")
#print(pdf_documents)

embedding_manager = EmbeddingManager()


#if __name__ == "__main__":
#    print("Hello, World!")