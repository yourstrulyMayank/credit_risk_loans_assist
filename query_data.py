import argparse
# from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM as Ollama
import os
from get_embedding_function import get_embedding_function
from logger_utils import setup_logger
logger = setup_logger()
# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


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

def query_rag_latest(query_text: str, db, model, latest_file):
    logger.info(f"Querying RAG with text: {query_text}")
    PROMPT_TEMPLATE = """
    You are an AI assistant. Your task is to answer the given question truthfully and strictly based on the provided context.

    Context:
    {context}

    ---
    
    Question: {question}
    
    File to search: {filename}
    
    Instructions:
    - If the answer is present in the context, provide it clearly and concisely.
    - If the answer is not found in the context, respond exactly with "Not Specified In The Document".
    - If the question asks for the company name, try to infer it somehow. Provide only the company name and no other words. If you can't infer, then say "Not Specified In The Document"
    
    """
    # Prepare the DB.
    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, filename=latest_file)
    # print(prompt)

    # model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    logger.info(f"Response: {formatted_response}")
    return response_text



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
