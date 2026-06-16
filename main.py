from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
from dotenv import load_dotenv
import os
import glob

load_dotenv()
hf_token=os.getenv("hf_token")
login(hf_token)

pdf=[f for f in glob.glob(os.path.join("PDF","*.pdf"))
    if f.endswith(".pdf")]

if not pdf:
    print("No PDFs found! Make sure files have .pdf extension!")
    exit()

non_pdf=[f for f in glob.glob(os.path.join("PDF","*"))
        if not f.endswith(".pdf")]

if non_pdf:
    print(f"Skipping non-PDF files:{[os.path.basename(f) for f in non_pdf]}")

print(f"Found {len(pdf)} PDF")

all_documents=[]
for pdf_path in pdf:
    docs=PyPDFLoader(pdf_path).load()
    all_documents.extend(docs)
    print(f"Loaded: {os.path.basename(pdf_path)} - {len(docs)} pages")

chunks=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
).split_documents(all_documents)

print(f"Total Chunks: {len(chunks)}")

embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

import shutil
if os.path.exists("./chroma_langchain_db"):
    shutil.rmtree("./chroma_langchain_db")
    print("Old Data cleared!")

vector_store=Chroma.from_documents(
    documents=chunks,
    collection_name="pdf-rag",
    embedding=embeddings,
    persist_directory="./chroma_langchain_db"
)

print(f"Done! Stored {vector_store._collection.count()} Vectors")