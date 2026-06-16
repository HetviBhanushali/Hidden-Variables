from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from huggingface_hub import login


load_dotenv()
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

embeddings = HuggingFaceEmbeddings(model = "BAAI/bge-small-en-v1.5")

vector_store = Chroma(
    collection_name= "pdf-rag",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings
)

print(f"Collection count: {vector_store._collection.count()}")

while True:
    user_prompt = input("Prompt (Enter 'q' to quit) : ")
    
    if user_prompt.split() == 'q':
        break
    
    results = vector_store.similarity_search(
        user_prompt, 
        k= 5
    )
    
for i, doc in enumerate(results, 1):
    print(f"Result: {i}")
    print(doc.page_content)
    print(" ")