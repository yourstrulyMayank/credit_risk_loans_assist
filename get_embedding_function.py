# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from logger_utils import setup_logger
from langchain_huggingface import HuggingFaceEmbeddings
logger = setup_logger()

# def get_embedding_function():
#     logger.info("✨ Getting Embedding Function")
#     # embeddings = BedrockEmbeddings(
#         # credentials_profile_name="default", region_name="us-east-1"
#     # )
#     embeddings = OllamaEmbeddings(model="llama3.2")
#     return embeddings




def get_embedding_function():
    logger.info("✨ Getting Embedding Function")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")