import os
import sys
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import populate_database
import clear_database
from get_embedding_function import get_embedding_function
from query_data import query_rag, query_rag_latest
from logger_utils import setup_logger
import asyncio
# from async_summary_pipeline import generate_summary_with_graph
from summary_utils import generate_summary

# ------------------- Config -------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'data/new'
CHROMA_PATH = "chroma"
PROMPTS_FILE_PATH = "utils/prompts.txt"
FILES_TRACK_PATH = "utils/files.txt"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logger = setup_logger()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Globals -------------------
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = OllamaLLM(model="llama3.2")

processing_status_upload = {"complete": False}
processing_status_fetch = {"complete": False}
fetched_results = {}
latest_file_data = {}

# ------------------- Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename:
        logger.info(f"Received file upload: {file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        logger.info(f"Saved file to: {filepath}")
        update_file_registry(file.filename)

        threading.Thread(target=run_populate_database, args=(file.filename,)).start()
        logger.info(f"Started background thread for: {file.filename}")
        return render_template('loading.html')
    return redirect(url_for('index'))


@app.route('/ask', methods=['GET', 'POST'])
def ask():
    document_titles = load_file_titles()
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            logger.info(f"Received question: {question}")
            response = query_rag(question, db, model)
            return render_template('ask.html', response=response, document_titles=document_titles)
    return render_template('ask.html', document_titles=document_titles)


@app.route('/batch_ask', methods=['POST'])
def batch_ask():
    questions = request.json.get('questions', [])
    logger.info(f"Received batch questions: {questions}")
    answers = [query_rag(q, db, model) for q in questions]
    return jsonify({"answers": answers})


@app.route('/clear_database', methods=['GET', 'POST'])
def clear_database_route():
    try:
        logger.info("Request to clear database received.")
        removed_files = clear_database.clear_database(db)
        sync_file_registry(removed_files)
        logger.info(f"Removed files: {removed_files}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/fetching_results', methods=['GET'])
def fetching_results():
    return render_template('fetching_results.html')


@app.route('/analyze', methods=['GET'])
def analyze():
    return render_template('analyze.html', data=fetched_results)


@app.route('/check_status_upload')
def check_status_upload():
    return jsonify({"complete": processing_status_upload["complete"]})


@app.route('/check_status_fetch')
def check_status_fetch():
    return jsonify({"complete": processing_status_fetch["complete"]})


# ------------------- Background Tasks -------------------
def run_populate_database(latest_filename):
    global processing_status_upload
    processing_status_upload["complete"] = False
    logger.info("Starting populate_database task...")
    try:
        populate_database.populate_database(db)
        logger.info("Database population complete.")
    finally:
        processing_status_upload["complete"] = True
        threading.Thread(target=run_query_database, args=(latest_filename,)).start()

def is_low_quality(response: str) -> bool:
    if not response:
        return True
    lowered = response.lower()
    return (
        "not specified" in lowered
        or len(response) < 15  # adjust threshold as needed
        or response.strip().endswith(":")  # might be an incomplete answer
    )


def run_query_database(latest_file):
    global fetched_results, processing_status_fetch
    processing_status_fetch["complete"] = False
    logger.info("Starting hybrid RAG and direct pass QA...")

    try:
        prompts = load_prompts(PROMPTS_FILE_PATH)

        docs = db.get(include=["metadatas", "documents"])
        file_chunks = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(docs["documents"], docs["metadatas"])
            if os.path.basename(meta.get("source", "")) == latest_file
        ]

        logger.info(f"Fetched {len(file_chunks)} chunks for file: {latest_file}")

        results = {}

        for question_name, question_text in prompts.items():
            try:
                logger.info(f"Processing question: {question_name}")

                # # ------- RAG Strategy ---------
                # rag_response = query_rag_latest(question_text, db, model, latest_file)

                # # Heuristic to detect poor answers
                # if is_low_quality(rag_response):
                #     logger.info(f"RAG answer for '{question_name}' seems low-quality. Using direct context.")

                #     # Fallback: Use direct context (top-N chunks from the current file)
                #     top_chunks_text = "\n---\n".join([doc.page_content for doc in file_chunks[:10]])
                #     prompt = f"{question_text}\n\nContext:\n{top_chunks_text}"
                #     fallback_answer = model.invoke(prompt).strip()
                #     results[question_name] = fallback_answer
                # else:
                #     results[question_name] = rag_response
                response = query_rag_latest(question_text, db, model, latest_file)
                results[question_name] = response

            except Exception as e:
                logger.error(f"Error processing question '{question_name}': {e}")
                results[question_name] = "[Error] " + str(e)

        # Add summary last
        summary = generate_summary(file_chunks, latest_file)
        results["Summary"] = summary
        refined_results = refine_answers_with_llm(results, prompts, model)
        fetched_results.update(refined_results)
        # fetched_results.update(results)
        processing_status_fetch["complete"] = True
        logger.info("All hybrid answers and summary generated.")

    except Exception as e:
        logger.error(f"Hybrid QA pipeline failed: {str(e)}")
        with open("data/error_log.txt", "w") as f:
            f.write(str(e))
        processing_status_fetch["complete"] = "error"


# ------------------- Helpers -------------------
def update_file_registry(filename):
    if not os.path.exists(FILES_TRACK_PATH):
        with open(FILES_TRACK_PATH, 'w'): pass
    with open(FILES_TRACK_PATH, 'r+') as f:
        logger.info(f"Updating file registry for: {filename}")
        lines = [line.strip().split(':')[0] for line in f.readlines()]
        if filename not in lines:
            f.write(f"{filename}:\n")

def refine_answers_with_llm(results: dict, prompts: dict, model, refine_prompt_path="utils/refine_prompt.txt") -> dict:
    """
    Refines the answers using an LLM and an external prompt template.
    
    Args:
        results (dict): Dictionary of {question_name: answer_from_rag}
        prompts (dict): Dictionary of {question_name: question_text}
        model: LLM instance with an `invoke(prompt)` method
        refine_prompt_path (str): Path to the refinement prompt template
    
    Returns:
        dict: Refined answers
    """
    try:
        with open(refine_prompt_path, "r") as f:
            prompt_template = f.read()
    except Exception as e:
        logger.error(f"Error loading refine prompt from {refine_prompt_path}: {e}")
        return {k: "[Refinement prompt missing]" for k in results}

    refined = {}

    for question_name, original_answer in results.items():
        if question_name == "Summary":
            refined[question_name] = original_answer
            continue

        try:
            question_text = prompts.get(question_name, "")
            prompt = prompt_template.format(
                context=original_answer,
                question=question_text
            )
            refined_answer = model.invoke(prompt).strip()
            refined[question_name] = refined_answer
        except Exception as e:
            logger.error(f"Error refining '{question_name}': {e}")
            refined[question_name] = "[Error during refinement] " + str(e)

    return refined


def sync_file_registry(removed_files):
    if not os.path.exists(FILES_TRACK_PATH):
        return
    with open(FILES_TRACK_PATH, 'r') as f:
        lines = f.readlines()
    with open(FILES_TRACK_PATH, 'w') as f:
        logger.info(f"Syncing registry, removing: {removed_files}")
        for line in lines:
            fname = line.strip().split(":")[0]
            if fname not in removed_files:
                f.write(line)

def load_file_titles():
    titles = []
    try:
        with open(FILES_TRACK_PATH, "r") as file:
            for line in file:
                key, _ = line.strip().split(":")
                titles.append(key)
    except FileNotFoundError:
        pass
    return titles


def load_prompts(file_path):
    prompts = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    prompts[key.strip()] = value.strip()
    return prompts


# ------------------- Main -------------------
if __name__ == '__main__':
    app.run(debug=True)
