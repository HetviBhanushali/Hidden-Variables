from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
import os, shutil                          # added shutil

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

pdf_path = os.path.join("PDF","vector.pdf")
loader = PyPDFLoader(os.path.join("PDF", "sample.pdf"))
documents = loader.load()
print("PDF LOADED!")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print("DOCUMENT SPLITTED")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("EMBEDDINGS LOADED")

# ✅ added - removes old data before saving new
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