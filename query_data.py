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

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from get_embedding_function import get_embedding_function

def query_rag_latest(query_text: str, db, model, latest_file: str):
    logger.info(f"Querying RAG with text: {query_text}")

    PROMPT_TEMPLATE = """
You are an intelligent assistant specialized in analyzing financial and loan documents.
Use only the provided context to answer the question below.

----------------
Context:
{context}
----------------

Instructions:
- Use only the context to answer.
- If the answer is clearly mentioned, extract and return it precisely.
- If the answer is not present, respond exactly with: Not Specified In The Document.
- For questions about company name, return only the name. If not found, say: Not Specified In The Document.

Question: {question}
"""

    # Step 1: Fetch all documents from DB and filter to just the latest file
    all_docs = db.get(include=["documents", "metadatas"])
    relevant_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if latest_file in meta.get("source", "")
    ]

    if not relevant_docs:
        logger.warning(f"No relevant chunks found for file: {latest_file}")
        return "Not Specified In The Document"

    logger.info(f"Found {len(relevant_docs)} chunks for file: {latest_file}")

    # Step 2: Embed query and chunks
    embed_fn = get_embedding_function()
    query_embedding = np.array(embed_fn.embed_query(query_text))
    chunk_embeddings = np.array([embed_fn.embed_query(doc.page_content) for doc in relevant_docs])

    # Step 3: Compute cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # Step 4: Max Marginal Relevance (MMR)
    def mmr(query_emb, doc_embs, k=5, lambda_param=0.5):
        selected = []
        selected_indices = set()

        while len(selected) < k and len(selected_indices) < len(doc_embs):
            mmr_score = -np.inf
            mmr_idx = -1
            for i, emb in enumerate(doc_embs):
                if i in selected_indices:
                    continue
                sim_to_query = cosine_similarity(query_emb, emb)
                sim_to_selected = max([cosine_similarity(emb, doc_embs[j]) for j in selected_indices], default=0)
                score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
                if score > mmr_score:
                    mmr_score = score
                    mmr_idx = i
            if mmr_idx == -1:
                break
            selected_indices.add(mmr_idx)
            selected.append(relevant_docs[mmr_idx])
        return selected

    top_chunks = mmr(query_embedding, chunk_embeddings, k=5)

    if not top_chunks:
        logger.warning("No relevant chunks found after manual MMR search.")
        return "Not Specified In The Document"

    # Step 5: Build the prompt and call the model
    context_text = "\n\n---\n\n".join([doc.page_content for doc in top_chunks])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text
    )

    try:
        response = model.invoke(prompt)
        logger.info(f"Response: {response}")
        return response.strip()
    except Exception as e:
        logger.error(f"Error invoking model: {e}")
        return "Error processing query."

# def query_rag_latest(query_text: str, db, model, latest_file: str) -> str:
#     from logger_utils import setup_logger
#     logger = setup_logger()
    
#     logger.info(f"Running full-context chunk-by-chunk search for: {query_text}")

#     PROMPT_TEMPLATE = """
# You are an intelligent assistant analyzing financial and loan documents.
# Answer the following question using only the context provided.
# If the answer is clearly mentioned, return it precisely.
# If the answer is not found in the context, say: Not Specified In The Document.

# Context:
# {context}

# Question: {question}
# """

#     try:
#         all_docs = db.get(include=["documents", "metadatas"])
#         file_chunks = [
#             Document(page_content=text, metadata=meta)
#             for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
#             if os.path.basename(meta.get("source", "")) == latest_file
#         ]

#         if not file_chunks:
#             logger.warning(f"No chunks found for file: {latest_file}")
#             return "Not Specified In The Document"

#         logger.info(f"Searching through {len(file_chunks)} chunks...")

#         for i, chunk in enumerate(file_chunks):
#             context_text = chunk.page_content
#             prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
#                 context=context_text,
#                 question=query_text
#             )
#             response = model.invoke(prompt).strip()
#             logger.debug(f"Chunk {i+1}/{len(file_chunks)} response: {response}")

#             if response and response.lower() != "not specified in the document":
#                 logger.info(f"Found answer in chunk {i+1}: {response}")
#                 return response

#         logger.info("No relevant answer found in any chunk.")
#         return "Not Specified In The Document"

#     except Exception as e:
#         logger.error(f"Error in query_rag_latest: {str(e)}")
#         return "[Error] " + str(e)




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
