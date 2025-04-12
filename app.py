import os
import sys
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

import populate_database
import clear_database
from get_embedding_function import get_embedding_function
from query_data import query_rag, query_rag_latest

# ------------------- Config -------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'data/new'
CHROMA_PATH = "chroma"
PROMPTS_FILE_PATH = "utils/prompts.txt"
FILES_TRACK_PATH = "utils/files.txt"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Globals -------------------
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = Ollama(model="llama3.2")

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        update_file_registry(file.filename)

        threading.Thread(target=run_populate_database, args=(file.filename,)).start()
        return render_template('loading.html')
    return redirect(url_for('index'))


@app.route('/ask', methods=['GET', 'POST'])
def ask():
    document_titles = load_file_titles()
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            response = query_rag(question, db, model)
            return render_template('ask.html', response=response, document_titles=document_titles)
    return render_template('ask.html', document_titles=document_titles)


@app.route('/batch_ask', methods=['POST'])
def batch_ask():
    questions = request.json.get('questions', [])
    answers = [query_rag(q, db, model) for q in questions]
    return jsonify({"answers": answers})


@app.route('/clear_database', methods=['GET', 'POST'])
def clear_database_route():
    try:
        removed_files = clear_database.clear_database(db)
        sync_file_registry(removed_files)
        return jsonify({"success": True})
    except Exception as e:
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
    try:
        populate_database.populate_database(db)
    finally:
        processing_status_upload["complete"] = True
        threading.Thread(target=run_query_database, args=(latest_filename,)).start()


def run_query_database(latest_file):
    global fetched_results, processing_status_fetch
    processing_status_fetch["complete"] = False

    # Load prompts
    prompts = load_prompts(PROMPTS_FILE_PATH)
    results = {k: query_rag_latest(v, db, model, latest_file) for k, v in prompts.items()}
    docs = db.get(include=["metadatas", "documents"])
    file_chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(docs["documents"], docs["metadatas"])
        if meta.get("source") == latest_file
    ]
    print(file_chunks)
    print('--------------')
    print(latest_file)
    summary = generate_summary(file_chunks, latest_file)
    results["Summary"] = summary

    # # Generate file-specific summary
    # summary_prompt = f"Provide a concise summary of the key financial insights in the file '{latest_file}'."
    # summary = query_rag_latest(summary_prompt, db, model, latest_file)
    # results["Summary"] = summary

    fetched_results.update(results)
    processing_status_fetch["complete"] = True


# ------------------- Helpers -------------------
def update_file_registry(filename):
    if not os.path.exists(FILES_TRACK_PATH):
        with open(FILES_TRACK_PATH, 'w'): pass
    with open(FILES_TRACK_PATH, 'r+') as f:
        lines = [line.strip().split(':')[0] for line in f.readlines()]
        if filename not in lines:
            f.write(f"{filename}:\n")


def sync_file_registry(removed_files):
    if not os.path.exists(FILES_TRACK_PATH):
        return
    with open(FILES_TRACK_PATH, 'r') as f:
        lines = f.readlines()
    with open(FILES_TRACK_PATH, 'w') as f:
        for line in lines:
            fname = line.strip().split(":")[0]
            if fname not in removed_files:
                f.write(line)

def generate_summary(chunks, file_name):
    llm = Ollama(model="llama3.2")
    chain = load_summarize_chain(llm, chain_type="stuff")
    file_chunks = [chunk for chunk in chunks if chunk.metadata.get("source") == file_name]
    return chain.invoke(file_chunks)

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
