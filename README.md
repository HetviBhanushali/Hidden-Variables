# PDF Question Answering Chatbot using RAG

## Overview

This project is a Retrieval-Augmented Generation (RAG) based PDF Question Answering Chatbot that allows users to upload PDF documents and ask questions about their contents.

Instead of relying solely on a Large Language Model (LLM), the system retrieves relevant information directly from uploaded documents and uses that context to generate accurate, document-grounded answers.

The chatbot also provides:

* Source citations (page references)
* Confidence scores
* Hallucination prevention
* Document summarization
* Performance evaluation metrics

---

## Features

### PDF Upload & Processing

* Upload one or more PDF files
* Extract text from PDFs
* OCR support for scanned PDFs
* Text cleaning and preprocessing

### Intelligent Retrieval

* Document chunking
* Embedding generation
* Vector database indexing
* Semantic similarity search

### Question Answering

* Natural language querying
* Context-aware answers
* Top-k retrieval
* Prompt engineering

### Explainability

* Source page references
* Confidence scores
* Answer validation

### Document Summarization

* Executive Summary
* Key Topics
* Main Conclusions

### Evaluation

* Retrieval Accuracy
* Response Latency
* Answer Relevance
* User Satisfaction

---

## System Architecture

```text
User Uploads PDF
        │
        ▼
PDF Processing
(Text Extraction + OCR)
        │
        ▼
Chunking
        │
        ▼
Embeddings
        │
        ▼
ChromaDB Vector Store
        │
        ▼
User Question
        │
        ▼
Similarity Search
        │
        ▼
Relevant Chunks
        │
        ▼
LLM (Gemini/Groq)
        │
        ▼
Generated Answer
        │
        ▼
Explainability Layer
(Source + Confidence)
        │
        ▼
Final Response
```

---

## Team Responsibilities

### Member 1 – PDF Processing & Chunking

Responsibilities:

* PDF upload
* Text extraction
* OCR support
* Text cleaning
* Chunk generation
* Page metadata extraction

Output:

```python
{
    "text": "...",
    "page": 12,
    "chunk_id": 45
}
```

---

### Member 2 – Embeddings & Vector Database

Responsibilities:

* Embedding generation
* ChromaDB setup
* Vector indexing
* Similarity search

Output:

```python
retrieved_chunks
similarity_scores
```

---

### Member 3 – Retrieval & LLM Pipeline

Responsibilities:

* Query handling
* Top-k retrieval
* Prompt engineering
* Gemini/Groq integration
* Answer generation

Output:

```python
{
    "question": "...",
    "answer": "..."
}
```

---

### Member 4 – Explainability & Evaluation

Responsibilities:

* Source highlighting
* Confidence scores
* Hallucination detection
* PDF summarization
* Evaluation metrics

Output:

```python
{
    "answer": "...",
    "source_page": 17,
    "confidence": "92%"
}
```

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### AI Frameworks

* LangChain
* Sentence Transformers

### LLM

* Gemini API / Groq API

### Vector Database

* ChromaDB

### PDF Processing

* PyMuPDF
* pdfplumber
* pytesseract (OCR)

---

## Project Structure

```text
pdf-rag-chatbot/

│
├── app.py
├── requirements.txt
├── README.md
│
├── uploads/
│
├── data/
│
├── vector_db/
│
├── modules/
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm_pipeline.py
│   ├── confidence.py
│   ├── validator.py
│   ├── summarizer.py
│   └── evaluation.py
│
└── tests/
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd pdf-rag-chatbot
```

### Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key

# OR

GROQ_API_KEY=your_api_key
```

---

## Running the Project

### Start Streamlit Application

```bash
streamlit run app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## Workflow

### Step 1

Upload PDF document.

### Step 2

System extracts text and creates chunks.

### Step 3

Chunks are converted into embeddings.

### Step 4

Embeddings are stored in ChromaDB.

### Step 5

User asks a question.

Example:

```text
What is CNN used for?
```

### Step 6

Retriever finds the most relevant chunks.

### Step 7

LLM generates an answer using retrieved context.

### Step 8

Explainability layer adds:

* Source page
* Confidence score
* Validation status

### Step 9

Final response displayed.

Example:

```text
Answer:
CNN is primarily used for image processing and computer vision.

Source:
Page 17

Confidence:
92%

Status:
Verified
```

---

## Evaluation Metrics

### Retrieval Accuracy

```text
Correct Retrievals / Total Queries
```

### Response Latency

```text
End Time - Start Time
```

### Answer Relevance

Scale:

```text
1 - Poor
2 - Fair
3 - Good
4 - Very Good
5 - Excellent
```

### User Satisfaction

```text
Helpful Responses / Total Responses
```

---

## Future Enhancements

* Multi-PDF support
* Chat history
* Hybrid search (BM25 + Vector Search)
* Dashboard analytics
* Multi-language PDFs
* Advanced citations
* Cloud deployment

---

## Expected Output

```text
Question:
What is CNN?

Answer:
CNN is a deep learning architecture used for image processing.

Source:
Page 17

Confidence:
92%

Status:
Verified
```

---

## Contributors

* Member 1 – PDF Processing & Chunking
* Member 2 – Embeddings & Vector Database
* Member 3 – Retrieval & LLM Pipeline
* Member 4 – Explainability & Evaluation
