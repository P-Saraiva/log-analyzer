import faiss
import numpy as np
from typing import List
from langchain.schema import Document

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)  # d = dimensão do embedding
        self.documents = []

    def add(self, vector, document: Document):
        self.index.add(np.array([vector]).astype("float32"))
        self.documents.append(document)

    def search(self, vector, k=5):
        D, I = self.index.search(np.array([vector]).astype("float32"), k)
        return [self.documents[i] for i in I[0]]

    def save(self, index_path="faiss.index", docs_path="documents.npy"):
        faiss.write_index(self.index, index_path)
        np.save(docs_path, np.array(self.documents, dtype=object))

    def load(self, index_path="faiss.index", docs_path="documents.npy"):
        self.index = faiss.read_index(index_path)
        self.documents = np.load(docs_path, allow_pickle=True).tolist()
