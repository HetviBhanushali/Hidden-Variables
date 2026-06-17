from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
import os, shutil, tempfile                

login(os.getenv("HF_TOKEN"))

def ingest_pdf(pdf_bytes):                     
    # new - save bytes as temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # same as before
    documents = PyPDFLoader(tmp_path).load()
    print("PDF LOADED!")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print("DOCUMENT SPLITTED")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("EMBEDDINGS LOADED")

    if os.path.exists("./chroma_langchain_db"):
        shutil.rmtree("./chroma_langchain_db")
        print("OLD DATA REMOVED!")

    vector_store = Chroma.from_documents(
        documents=chunks,
        collection_name="pdf-rag",
        embedding=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    print("SAVED!")

    os.unlink(tmp_path)                       