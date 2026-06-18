# PDF Question-Answering using Retrieval-Augmented Generation (RAG)

## Overview
PDF Question-Answering using Retrieval-Augmented Generation (RAG) is an intelligent document analysis system that allows users to upload PDF documents and ask questions in natural language. The system retrieves relevant information from the uploaded document and generates accurate, context-aware answers using semantic search and a Large Language Model (LLM).

This project combines **LangChain**, **Chroma Vector Database**, **Hugging Face Embeddings**, **Groq API**, and **Llama 3.1** to build an efficient document-based question-answering pipeline.

---

## Features
- Upload PDF documents through a web interface
- Extract text from PDFs automatically
- Split documents into semantic chunks
- Generate vector embeddings using Hugging Face models
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Generate context-aware answers using LLM
- Interactive chatbot interface using Streamlit
- Reduced hallucinations using RAG architecture

---

## Problem Statement
Searching for specific information inside large PDF documents such as research papers, manuals, and reports is time-consuming and inefficient. Traditional keyword-based search methods often fail to understand the context of user queries, resulting in inaccurate results.

This project solves this problem by enabling users to interact with PDF documents using natural language questions and receiving accurate answers grounded in document content.

---

## System Architecture

```text
User Uploads PDF
       ↓
PDF Processing
       ↓
Text Chunking
       ↓
Embedding Generation
       ↓
Chroma Vector Database
       ↓
User Query
       ↓
Semantic Retrieval
       ↓
LLM (Llama 3.1 via Groq)
       ↓
Generated Answer
```

---

## Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- LangChain

### Database
- Chroma Vector Database

### AI / ML Models
- Hugging Face Embeddings  
  - BAAI/bge-small-en-v1.5
- Llama 3.1 8B Instant

### APIs
- Groq API
- Hugging Face Hub

---

## Modules

### 1. PDF Processing Module
- Upload PDF files
- Extract text using PyPDFLoader

### 2. Text Chunking Module
- Split documents into chunks
- Maintain contextual overlap

### 3. Embedding Module
- Convert chunks into vector embeddings

### 4. Vector Database Module
- Store embeddings in ChromaDB
- Perform similarity search

### 5. Question Answering Module
- Retrieve relevant chunks
- Send context to LLM
- Generate final answer

---

## Installations

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Usage
1. Launch application
2. Upload PDF document
3. Wait for indexing
4. Ask questions in chat
5. Receive context-aware answers

---

## Results
- Successfully indexed PDF documents
- Efficient semantic retrieval using ChromaDB
- Accurate context-based answer generation
- Reduced hallucination using RAG architecture
- User-friendly chat interface

---

## Limitations
- Supports only text-based PDFs
- Scanned PDFs require OCR
- Large documents increase indexing time
- Single PDF processing at a time
- Requires internet for Groq API

---

## Future Improvements
- OCR support for scanned PDFs
- Multi-PDF querying
- Conversation memory
- Support DOCX, PPT, TXT
- Advanced reranking
- Cloud deployment

---

## References
1. LangChain Documentation  
2. Chroma Documentation  
3. Hugging Face Model Hub  
4. Groq Documentation  
5. Streamlit Documentation  
6. RAG Research Paper (NeurIPS 2020)

---

## Author
Developed as part of an academic mini-project on **AI-powered Document Intelligence using Retrieval-Augmented Generation (RAG)**.