from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import os

DB_PATH = "./chroma_langchain_db"

def build_index(pdf_path):

    # Delete old DB if it exists
   # if os.path.exists(DB_PATH):
    #    shutil.rmtree(DB_PATH)

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Create Chroma DB
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf-rag",
        persist_directory=DB_PATH
    )

    return len(chunks)