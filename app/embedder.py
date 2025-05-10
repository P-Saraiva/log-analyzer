# app/embedder.py

from typing import List
import requests
import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_URL= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class Embedder:
    def __init__(self, model_url: str = f"{OLLAMA_URL}/api/embeddings", model_name: str = "nomic-embed-text"):
        self.model_url = model_url
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos usando Ollama (ou outra API compatível).
        """
        embeddings = []

        for text in texts:
            try:
                #Send request to the model API
                response = requests.post(self.model_url, json={
                    "model": self.model_name,
                    "prompt": text
                })
                
                response.raise_for_status()
                data = response.json()

                if "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    logger.warning(f"Nenhum embedding retornado para o texto: {text[:30]}...")

            except Exception as e:
                logger.error(f"Erro ao gerar embedding: {e}")
                embeddings.append([])  # Placeholder vazio para evitar falhas

        return embeddings
