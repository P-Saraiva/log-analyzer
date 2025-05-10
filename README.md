# RAG-Based Log Analyzer

This is an AI-driven system that analyzes application logs using Retrieval-Augmented Generation (RAG). It uses embeddings to index logs, retrieves relevant sections based on user queries, and generates intelligent responses using a Large Language Model (LLM).

## 🔍 Use Case

- Upload or stream logs (e.g., syslog, app logs)
- Index log entries into a vector database
- Query the system using natural language (e.g., "Why did this error happen?")
- Get LLM-generated analysis and suggestions based on relevant log context

## 🧱 Architecture

- Python (FastAPI for backend)
- FAISS as local vector database
- Sentence Transformers for embeddings
- OpenAI or local LLM for response generation

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional but recommended)

### 1. Clone the repo

```bash
git clone https://github.com/p-saraiva/rag-log-analyzer.git
cd rag-log-analyzer
