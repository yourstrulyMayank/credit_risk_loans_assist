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


CHROMA_PATH = "chroma"
AVAILABLE_FILES_PATH = "utils\\files.txt"

def main():    
    return clear_database()

def clear_database(db):
    try:
        print("✨ Clearing Database")
        ids_to_delete = db.get()["ids"]
        print(ids_to_delete)        
        db.delete(ids=ids_to_delete)
        # db._client._system.stop()
        # SharedSystemClient._identifer_to_system.pop(vector_db._client._identifier, None)
        # db = None
        # if os.path.exists(CHROMA_PATH):
        #     shutil.rmtree(CHROMA_PATH)
        with open(AVAILABLE_FILES_PATH, "w") as file:
            file.write("")
        return True
    except Exception as e:
        print(e)
        return False
    

if __name__ == "__main__":
    main()
