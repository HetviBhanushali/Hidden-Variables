from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

vector_store = Chroma(
    collection_name="pdf-rag",
    persist_directory = "./chroma_langchain_db",
    embedding_function = embeddings,
)
print(f"Collection count: {vector_store._collection.count()}")

while True:
    user_prompt = input("Prompt (type 'q' to quit):")
    if user_prompt.strip() == "q":
        break

    results = vector_store.similarity_search_with_score(
        user_prompt,
        k = 5
    )
    score_result = []
    for doc, score in results:
        if score<1.0:
            score_result.append((doc,score))

    if not score_result:
        print("No relevant information found in the PDF.")
        continue

    for i, doc in enumerate(score_result, 1):
        print(f"Result: {i}")
        print(f"Source : {doc.metadata.get('filename')}")
        print(f"Page : {doc.metadata.get('page')}")
        print(doc.page_content)
        print(" ")