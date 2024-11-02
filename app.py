import os 
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API
openai_key = os.getenv('OPENAI_API_KEY')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)
# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)
