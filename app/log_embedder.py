import requests
import os
import app.main
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.faiss_store import FAISSStore

embedder_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class EmbedRequest:
    def __init__(self, file: str):
        self.file = file

    def to_dict(self):
        return {"file": self.file}

def embed_logs_from_file(file_path):
    req = EmbedRequest(file=file_path)
    
    print(f"Indexando logs do arquivo: {file_path}")
    
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Total de linhas lidas: {len(lines)}")
    
    for i, line in enumerate(lines):
        embedding = app.main.embedder.embed_texts([line])[0]
        doc = Document(page_content=line)
        app.main.faiss_store.add(embedding, doc)
        if i % 50 == 0:
            print(f"Indexadas {i}/{len(lines)} linhas...")

    print(f"Indexação concluída: {len(lines)} linhas inseridas no FAISS.")