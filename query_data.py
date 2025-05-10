import argparse
# from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM as Ollama
import os
from get_embedding_function import get_embedding_function
from logger_utils import setup_logger
logger = setup_logger()
AVAILABLE_FILES_PATH = "utils\\files.txt"


def main():
    # Create CLI.    

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str, db, model):
    logger.info(f"Querying RAG with text: {query_text}")
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    # Prepare the DB.
    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    logger.info(f"Response: {formatted_response}")
    return response_text

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from get_embedding_function import get_embedding_function

# def query_rag_latest(query_text: str, db, model, latest_file: str):
#     logger.info(f"Querying RAG with text: {query_text}")

#     PROMPT_TEMPLATE = """
# You are an intelligent assistant specialized in analyzing financial and loan documents.
# Use only the provided context to answer the question below.

# ----------------
# Context:
# {context}
# ----------------

# Instructions:
# - Use only the context to answer.
# - If the answer is clearly mentioned, extract and return it precisely.
# - If the answer is not present, respond exactly with: Not Specified In The Document.
# - For questions about company name, return only the name. If not found, say: Not Specified In The Document.

# Question: {question}
# """

#     # Step 1: Retrieve top-k results using similarity search
#     all_results = db.similarity_search_with_score(query_text, k=20)

#     # Step 2: Filter results to match the latest file
#     filtered_results = [
#         (doc, score) for doc, score in all_results
#         if latest_file in doc.metadata.get("source", "")
#     ][:5]  # Take top 5 relevant chunks for this file

#     if not filtered_results:
#         logger.warning(f"No relevant chunks found for file: {latest_file}")
#         return "Not Specified In The Document"

#     logger.info(f"Found {len(filtered_results)} relevant chunks for file: {latest_file}")

#     # Step 3: Build the prompt and call the model
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])
#     prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
#         context=context_text,
#         question=query_text
#     )

#     try:
#         response = model.invoke(prompt)
#         logger.info(f"Response: {response}")
#         return response.strip()
#     except Exception as e:
#         logger.error(f"Error invoking model: {e}")
#         return "Error processing query."

def query_rag_latest(query_text: str, db, model, latest_file: str):
    logger.info(f"Querying RAG with text: {query_text}")

    all_results = db.similarity_search_with_score(query_text, k=20)

    filtered_results = [
        (doc, score) for doc, score in all_results
        if latest_file in doc.metadata.get("source", "")
    ][:5]

    if not filtered_results:
        logger.warning(f"No relevant chunks found for file: {latest_file}")
        return "Not Specified In The Document"

    logger.info(f"Found {len(filtered_results)} relevant chunks for file: {latest_file}")

    context_text = "\n\n".join([doc.page_content for doc, _ in filtered_results])
    combined_input = f"Question: {query_text}\n\nContext:\n{context_text}"

    try:
        response = model.invoke(combined_input)
        logger.info(f"Response: {response}")
        return response.strip()
    except Exception as e:
        logger.error(f"Error invoking model: {e}")
        return "Error processing query."


def get_latest_file():
    logger.info("Fetching the latest file from files.txt")
    """
    Read the last line from files.txt to get the most recent file.
    """
    if os.path.exists(AVAILABLE_FILES_PATH):
        with open(AVAILABLE_FILES_PATH, "r") as file:
            lines = file.readlines()
            if lines:
                # Extract the latest file from the last line
                last_line = lines[-1].strip()
                file_name, _ = last_line.split(":")
                logger.info(f"Latest file: {file_name}")
                return file_name
    return None

if __name__ == "__main__":
    main()
