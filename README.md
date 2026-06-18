# Hidden Variables 🔍

A **PDF Question-Answering Chatbot** built with Retrieval-Augmented Generation (RAG). Upload any PDF and ask questions — the system retrieves relevant chunks from the document and generates accurate, context-grounded answers using a Groq-powered LLM.

---

## How It Works

```
Upload PDF
    │
    ▼
PyPDFLoader → Text Extraction
    │
    ▼
RecursiveCharacterTextSplitter → Chunks (1000 tokens, 200 overlap)
    │
    ▼
HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
    │
    ▼
ChromaDB Vector Store (persisted locally)
    │
    ▼
User Question → Similarity Search (Top-k=3)
    │
    ▼
Groq LLM (Llama 3.1 8B Instant) → Answer
    │
    ▼
Streamlit UI → Response Displayed
```

---

## Project Structure

```
Hidden-Variables/
│
├── streamlit_ui.py      # Streamlit frontend — PDF upload + chat interface
├── main.py              # LLM pipeline — question answering via Groq
├── vector.py            # PDF loading, chunking, and ChromaDB indexing
├── vectorization.py     # Embedding model + similarity search (retrieve)
├── basic.py             # (utility / scratch module)
├── requirement.txt      # Python dependencies
└── README.md
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace — `BAAI/bge-small-en-v1.5` |
| Vector DB | ChromaDB (local persistence) |
| PDF Loading | LangChain `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Auth | `python-dotenv`, `huggingface_hub` login |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/HetviBhanushali/Hidden-Variables.git
cd Hidden-Variables
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

- Get your Groq API key at [console.groq.com](https://console.groq.com)
- Get your HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## Running the App

```bash
streamlit run streamlit_ui.py
```

Then open your browser at `http://localhost:8501`.

---

## Usage

1. Upload a PDF using the sidebar file uploader.
2. Wait for the success message — the PDF is chunked and indexed into ChromaDB.
3. Type your question in the chat input.
4. The app retrieves the top 3 most relevant chunks and passes them to the LLM.
5. If the answer isn't found in the document, the model responds: *"I could not find the answer in the provided PDF."*

---

## Key Implementation Details

- **Chunking:** `chunk_size=1000`, `chunk_overlap=200` via `RecursiveCharacterTextSplitter`
- **Embeddings:** `BAAI/bge-small-en-v1.5` via `langchain-huggingface`
- **Vector store:** ChromaDB persisted at `./chroma_langchain_db`, keyed by PDF filename (without extension)
- **Retrieval:** Top-3 similarity search (`k=3`)
- **LLM prompt:** Strictly context-grounded — the model is instructed not to hallucinate beyond the provided chunks

---

## Contributors

This is a collaborative student project — **Hidden Variables** — built as part of a RAG exploration exercise.

| Module | Responsibility |
|---|---|
| `vector.py` | PDF loading, chunking, ChromaDB indexing |
| `vectorization.py` | Embedding model, similarity search |
| `main.py` | Groq LLM integration, prompt engineering, answer generation |
| `streamlit_ui.py` | Frontend UI, session state management |

---

## Future Scope

- Multi-PDF support with collection switching
- Explainability layer — source page references + confidence scores
- Chat history / conversation memory
- Evaluation metrics (retrieval accuracy, answer relevance)
- Hybrid search (BM25 + vector)
- Cloud deployment (Streamlit Cloud / HuggingFace Spaces)