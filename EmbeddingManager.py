from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import numpy as np
import chromadb
import uuid
from sklearn.metrics.pairwise import cosine_similarity



class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
        

    def _load_model(self):
        try:
            print(f"Loading model {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully. Embedding size: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e
        
    def generate_embeddings(self, text: List[str]) -> List[np.ndarray]:
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(text)} texts")
        
        embeddings = self.model.encode(text, show_progress_bar=True)
        print(f"Embeddings generated successfully. Shape: {embeddings.shape}")
        return embeddings