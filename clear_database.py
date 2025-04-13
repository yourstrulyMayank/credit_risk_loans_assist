import argparse
import os
import shutil
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
# from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from logger_utils import setup_logger
logger = setup_logger()


CHROMA_PATH = "chroma"
AVAILABLE_FILES_PATH = "utils\\files.txt"

def main():    
    return clear_database()

def clear_database(db):
    try:
        logger.info("✨ Clearing Database")
        ids_to_delete = db.get()["ids"]
        logger.info(f'ids_to_delete: {ids_to_delete}')
        if not ids_to_delete:
            logger.info("No IDs to delete.")
            return False        
        db.delete(ids=ids_to_delete)
        # db._client._system.stop()
        # SharedSystemClient._identifer_to_system.pop(vector_db._client._identifier, None)
        # db = None
        # if os.path.exists(CHROMA_PATH):
        #     shutil.rmtree(CHROMA_PATH)
        with open(AVAILABLE_FILES_PATH, "w") as file:
            file.write("")
            logger.info(f"Cleared {AVAILABLE_FILES_PATH} file.")
        logger.info("✨ Database cleared successfully.")
        return True
    except Exception as e:
        print(e)
        return False
    

if __name__ == "__main__":
    main()
