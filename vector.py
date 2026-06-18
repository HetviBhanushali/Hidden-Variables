# Import required libraries

# ChromaDB vector database
from langchain_chroma import Chroma

# Hugging Face embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# PDF loader to extract text from PDF files
from langchain_community.document_loaders import PyPDFLoader

# Text splitter for dividing large text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Path where ChromaDB will store vector embeddings
DB_PATH = "./chroma_langchain_db"


# Function to create and store vector embeddings from a PDF
def build_index(pdf_path, collection_name):

    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into smaller chunks
    # chunk_size = maximum characters per chunk
    # chunk_overlap = common characters between consecutive chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Create chunks from the loaded document
    chunks = splitter.split_documents(documents)

    # Load Hugging Face embedding model
    # Converts text chunks into numerical vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Store document chunks and embeddings in ChromaDB
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_PATH
    )

    # Display collection creation message
    print("Created collection:", collection_name)

    # Return total number of chunks created
    return len(chunks)