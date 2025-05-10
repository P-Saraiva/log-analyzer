import faiss
import numpy as np
from typing import List

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vector: List[float], text: str):
        vector_np = np.array([vector]).astype("float32")
        self.index.add(vector_np)
        self.texts.append(text)

    def search(self, query_vector: List[float], k: int = 5): # K = Number of nearest neighbors to return
        query_np = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_np, k)
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        return results
