import os
from typing import List
import chromadb
from langchain_core.documents import Document
import numpy as np
import uuid
from datetime import datetime


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
        
    def _initialize_store(self):
        try:
            print(f"Initializing vector store in {self.persist_directory}")
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "Vector store for PDF documents"})
            print(f"Vector store {self.collection_name} initialized in {self.persist_directory}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise e

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")
        
        ids = []
        metadata = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(id)
            
            metadata = dict(doc.metadata)
            metadata['document_id'] = id
            metadata['content_length'] = len(doc.page_content)
            metadata['source'] = doc.metadata.get('source', 'unknown')
            metadata['created_at'] = datetime.now().isoformat()
            metadata['updated_at'] = datetime.now().isoformat()
            metadata['embedding_size'] = embedding.shape[0]
            metadata['embedding_type'] = 'float32'
            metadata['embedding_format'] = 'numpy'
            metadata['embedding_shape'] = embedding.shape
            
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        try:
            print(f"Adding {len(documents)} documents to vector store")
            self.collection.add(
                documents=documents_text,
                embeddings=embeddings_list,
                ids=ids,
                metadata=metadata
            )
            print(f"Added {len(documents)} documents to vector store")
            print(f"Vector store contains {self.collection.count()} documents")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise e
            
    def query(self, query: str):
        return True
        #TODO: Implement query method