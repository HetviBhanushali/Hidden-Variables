from langchain_groq import ChatGroq
#from langchain_huggingface import ChatHuggingFace
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("APIKEY")
#HF_TOKEN = os.getenv("HF_TOKEN")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    groq_api_key=API_KEY
)

"""llm = ChatHuggingFace(
    model = "BAAI/bge-small-en-v1.5",
    hugging_face=HF_TOKEN
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    gemini_api_key=GOOGLE_API_KEY 
)

llm = ChatOpenAI(
    model = "gpt-4o",
    openai_api_key=OPENAI_API_KEY
)"""

response = llm.invoke("Hello")
print(response.content)

