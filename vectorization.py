from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

def retrieve(query, k=5):

    vector_store = Chroma(
        collection_name="pdf-rag",
        persist_directory="./chroma_langchain_db",
        embedding_function=embeddings
    )

    return vector_store.similarity_search(
        query,
        k=k
    )