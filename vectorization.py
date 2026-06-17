from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma(
    collection_name="pdf-rag",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
)

print(f"Collection count: {vector_store._collection.count()}")


def retrieve(query):

    results = vector_store.similarity_search_with_score(
        query,
        k=5
    )

    filtered_results = [
        (doc, score)
        for doc, score in results
        if score < 1.0
    ]

    return filtered_results


while True:

    user_prompt = input("\nPrompt (type 'q' to quit): ")

    if user_prompt.strip().lower() == "q":
        break

    filtered_results = retrieve(user_prompt)

    if not filtered_results:
        print("No relevant information found in the PDF.")
        continue

    for i, (doc, score) in enumerate(filtered_results, start=1):
        print(f"\nResult {i}")
        print(f"Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('filename', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(doc.page_content)