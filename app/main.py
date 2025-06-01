# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import requests

from app.embedder import Embedder
from app.faiss_store import FAISSStore
from app.log_embedder import embed_logs_from_file
#from .retriever import retrieve_relevant_chunks
#from .responder import generate_response

LOGS_DIR = os.environ.get("LOGS_DIR", "/logs")
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(title="RAG-Based Log Analyzer")
embedder = Embedder()
faiss_store = FAISSStore(dim=768)  # Dimensão do embedding, ajuste conforme o modelo ()

class EmbedRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    
    
@app.post("/embed")
def generate_embeddings(payload: dict):
    texts = payload.get("texts", [])
    vectors = embedder.embed_texts(texts)
    return {"vectors": vectors}

@app.post("/add")
def add_log(req: EmbedRequest):
    embedding_result = embedder.embed_texts([req.text])
    vector = embedding_result[0]
    faiss_store.add(vector, req.text)
    return {"status": "ok", "msg": "added"}

@app.post("/search")
def search_log(req: SearchRequest):
    embedding_result = embedder.embed_texts([req.query])
    query_vector = embedding_result[0]
    results = faiss_store.search(query_vector, k=req.top_k)
    return {"matches": results}


@app.post("/upload_logs")
async def upload_logs(file: UploadFile = File(...)):
    file_location = os.path.join(LOGS_DIR, file.filename)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        # Indexa logs no FAISS
        embed_logs_from_file(file_location)
        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query")
# def query_logs(req: QueryRequest):
#     try:
#         # Recupera contexto relevante dos logs
#         relevant_chunks = retrieve_relevant_chunks(req.question)
        
#         # Gera resposta com modelo Ollama
#         answer = generate_response(req.question, relevant_chunks)
        
#         return JSONResponse(content={
#             "question": req.question,
#             "answer": answer,
#             "context": relevant_chunks
#         })
#
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))
