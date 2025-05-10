import argparse
import os
import shutil
import filetype
from PyPDF2 import PdfReader
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
# from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from load_images import process_images_to_pdf
from logger_utils import setup_logger
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
logger = setup_logger()


CHROMA_PATH = "chroma"
DATA_PATH = "data"
NEW_DATA_PATH = "data\\new"
AVAILABLE_FILES_PATH = "utils\\files.txt"

def main():
    return populate_database()

def populate_database(db):  
    # Load new documents
    logger.info("Loading new documents...")
    documents = load_documents()
    
    if not documents:
        print("No new documents found.")
        logger.info("No new documents found.")
        return

    # Get the source filenames from the new documents
    new_sources = {doc.metadata['source'].split('\\')[-1] for doc in documents}
    logger.info(f"Found new sources: {new_sources}")
    # Remove old entries from the database
    logger.info("Removing old entries for re-uploaded documents...")
    print("Removing old entries for re-uploaded documents...")
    remove_existing_documents(db, new_sources)

    # Split and add new chunks
    logger.info("Splitting documents into chunks...")
    chunks = split_documents(documents)    
    print("Adding to Database")
    logger.info("Adding chunks to database...")
    add_to_chroma(chunks, db)
    print("Added to Database")

    # Update file list
    add_file_to_list(db, documents[-1].metadata['source'].split('\\')[-1], len(chunks))    
    print("Added file to list")
    logger.info("Moving processed files to permanent storage...")
    # Move processed files to permanent storage
    for filename in os.listdir(NEW_DATA_PATH):
        src = os.path.join(NEW_DATA_PATH, filename)
        dest = os.path.join(DATA_PATH, filename)
        shutil.move(src, dest)
    print(f"All files moved from {NEW_DATA_PATH} to {DATA_PATH}")
    logger.info(f"All files moved from {NEW_DATA_PATH} to {DATA_PATH}")



def load_documents():
    document_loader = PyPDFDirectoryLoader(NEW_DATA_PATH)
    
    for filename in os.listdir(NEW_DATA_PATH):
        logger.info("Checking and processing files in NEW_DATA_PATH...")
        file_path = os.path.join(NEW_DATA_PATH, filename)
        
        # Check file type
        if filename.endswith('.pdf'):
            if is_text_pdf(file_path):
                logger.info(f"Text-based PDF detected: {filename}. No further processing required.")
            else:
                logger.info(f"Image-based PDF detected: {filename}. Processing with OCR.")
                process_images_to_pdf(file_path)
                os.remove(file_path)  # Remove the original image-based PDF after processing
        elif is_image_file(file_path):
            logger.info(f"Image file detected: {filename}. Processing with OCR.")
            process_images_to_pdf(file_path)
            os.remove(file_path)  # Remove the original image after processing

    # After processing, load all documents from NEW_DATA_PATH
    logger.info("Loading all text-based PDFs in NEW_DATA_PATH.")
    return document_loader.load()


def is_text_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    return True
        return False
    except Exception as e:
        logger.info(f"Error reading PDF {file_path}: {e}")
        return False

def is_image_file(file_path):
    kind = filetype.guess(file_path)
    return kind and kind.mime.startswith("image/")


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", " "],  # Preserve table rows
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    logger.info(f"Splitting {len(documents)} documents in parallel...")
    def split_single_document(doc):
        return text_splitter.split_documents([doc])

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(split_single_document, documents))
    # Flatten the list of lists
    total_chunks = [chunk for sublist in results for chunk in sublist]
    logger.info(f"Total chunks created: {len(total_chunks)}")
    
    return total_chunks


# def add_to_chroma(chunks: list[Document], db):
#     # Load the existing database.
#     # db = Chroma(
#     #     persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
#     # )

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)

#     # Add or Update the documents.
#     existing_items = db.get(include=[])  # IDs are always included by default
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         # db.persist()
#     else:
#         print("✅ No new documents to add")
def add_to_chroma(chunks: list[Document], db, batch_size=50):
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if not new_chunks:
        logger.info("✅ No new documents to add")
        return

    logger.info(f"👉 Adding {len(new_chunks)} new documents to DB in batches")

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        db.add_documents(batch, ids=batch_ids)
        logger.info(f"✅ Added batch {i // batch_size + 1} of {len(batch)} chunks")

    logger.info("✅ All new documents added")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_file_to_list(db, file_name, new_chunk_count):
    print("Here in adding file to list")
    """
    Add the file name and chunk count difference to the available files list.

    :param db: Chroma database instance.
    :param file_name: Name of the file being processed.
    :param new_chunk_count: Number of chunks generated for the file.
    """
    # Fetch existing chunks for the file from the database.
    logger.info(f"Updating {AVAILABLE_FILES_PATH} with file: {file_name}")
    existing_items = db.get(include=["metadatas", "documents"])  # Use "metadatas"
    # print(existing_items)
    existing_chunks = [
    doc for doc, metadata in zip(existing_items["documents"], existing_items["metadatas"]) 
    if metadata["source"] == file_name]


    # Count existing chunks for the given file.
    existing_chunk_count = len(existing_chunks)
    chunk_difference = new_chunk_count - existing_chunk_count

     # Read current lines and filter out lines for the file
    if os.path.exists(AVAILABLE_FILES_PATH):
        with open(AVAILABLE_FILES_PATH, "r") as file:
            lines = file.readlines()
        lines = [line for line in lines if not line.startswith(f"{file_name}:")]
    else:
        lines = []

    # Append the new entry
    lines.append(f"{file_name}:{chunk_difference}\n")

    # Write back to the file
    with open(AVAILABLE_FILES_PATH, "w") as file:
        file.writelines(lines)

    # # Append the new file name and chunk count to the file list.
    # with open(AVAILABLE_FILES_PATH, "a") as file:
    #     file.write(f"{file_name}:{chunk_difference}\n")
    logger.info(f"✅ Updated file list for {file_name} with {chunk_difference} new chunks.")
    print(f"✅ Updated file list for {file_name} with {chunk_difference} new chunks.")

def remove_existing_documents(db, sources):
    """
    Remove all chunks related to the given source files from the Chroma database.

    :param db: Chroma database instance.
    :param sources: Set of filenames (source) to remove.
    """
    existing_items = db.get(include=["metadatas"])  # Get metadata from the DB
    existing_ids = [
        item_id for item_id, metadata in zip(existing_items["ids"], existing_items["metadatas"]) 
        if metadata["source"].split('\\')[-1] in sources
    ]

    if existing_ids:
        logger.info(f"Deleting {len(existing_ids)} existing entries from the database...")
        db.delete(existing_ids)
        logger.info("Old entries removed successfully.")
        # Remove from file list
        if os.path.exists(AVAILABLE_FILES_PATH):
            with open(AVAILABLE_FILES_PATH, "r") as file:
                lines = file.readlines()
            new_lines = [
                line for line in lines if line.split(":")[0] not in sources
            ]
            with open(AVAILABLE_FILES_PATH, "w") as file:
                file.writelines(new_lines)
            logger.info(f"Removed entries from {AVAILABLE_FILES_PATH}")
    else:
        logger.info("No existing entries found for these documents.")


if __name__ == "__main__":
    main()
