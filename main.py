from langchain_groq import ChatGroq
from dotenv import load_dotenv
from huggingface_hub import login
from vectorization import retrieve
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

login(HF_TOKEN)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=API_KEY
)

def answer_question(question):

    # Retrieve relevant chunks
    docs = retrieve(question, k=3)

    print("Retrieved Docs:", len(docs))

    # Debug: print retrieved content
    for i, d in enumerate(docs, start=1):
        print(f"\nDocument {i}")
        print(d.page_content[:300])

    # Combine chunks into context
    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    print("\nContext:")
    print(context)

    # Handle empty retrieval
    if not context.strip():
        return "No relevant content found in the vector database."

    prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
'I could not find the answer in the provided PDF.'

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content