from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import os
from utils import split_documents
from EmbeddingManager import EmbeddingManager
from VectorStore import VectorStore
from RAGRetriever import RAGRetriever

document = Document(page_content="Hello, World!",
                    metadata={"source": "test.txt"})

embedding_manager = EmbeddingManager()
vector_store = VectorStore()

pdf_loader = DirectoryLoader(
    path="./PDFs",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)
pdf_documents = pdf_loader.load()

chunks =split_documents(pdf_documents)

text = [chunks.page_content for chunks in chunks]

embeddings = embedding_manager.generate_embeddings(text)

vector_store.add_documents(chunks, embeddings)

rag_retriever = RAGRetriever(vector_store, embedding_manager)

print(rag_retriever.retrieve("What is attention is all you need?"))


#if __name__ == "__main__":
#    print("Hello, World!")
