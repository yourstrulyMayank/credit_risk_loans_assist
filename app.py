import os
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify
import populate_database
import clear_database
from query_data import query_rag, query_rag_latest
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama

app = Flask(__name__)
UPLOAD_FOLDER = 'data\\new'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CHROMA_PATH = "chroma"
PROMPTS_FILE_PATH = "utils//prompts.txt"
processing_status_upload = {"complete": False}
processing_status_fetch = {"complete": False}



embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = Ollama(model="llama3.2")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run populate_database.py in the background
        threading.Thread(target=run_populate_database).start()
        return render_template('loading.html')  # Show loading page while population happens


@app.route('/ask', methods=['GET', 'POST'])
def ask():
    document_titles = load_file_titles()
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            # Process question using query_rag
            response = query_rag(question, db, model)  # Replace with actual RAG logic
            return render_template('ask.html', response=response, document_titles=document_titles)
    return render_template('ask.html', document_titles=document_titles)


@app.route('/batch_ask', methods=['POST'])
def batch_ask():
    data = request.json
    questions = data.get('questions', [])
    answers = []

    # Process each question and append the response
    for question in questions:
        answer = query_rag(question, db, model)  # Replace with actual RAG logic
        answers.append(answer)

    return jsonify({"answers": answers})


@app.route('/clear_database', methods=['GET','POST'])
def clear_database_route():    
    
    try:               
        if clear_database.clear_database(db):
            # embedding_function = get_embedding_function()
            # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            # model = Ollama(model="mistral")
            return jsonify({"success": True})
        else:
            return jsonify({"success": False})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def restart_flask_app():
    """Restart the Flask application programmatically."""
    print("Restarting Flask application...")
    os.execv(sys.executable, ['python'] + sys.argv)


def run_populate_database():
    global processing_status_upload
    processing_status_upload["complete"] = False
    try:
        populate_database.populate_database(db)
    finally:
        processing_status_upload["complete"] = True
        # After database population, redirect to fetching results
        threading.Thread(target=redirect_to_fetching_results).start()


@app.route('/check_status_upload', methods=['GET'])
def check_status_upload():
    return jsonify({"complete": processing_status_upload["complete"]})


@app.route('/check_status_fetch', methods=['GET'])
def check_status_fetch():
    return jsonify({"complete": processing_status_fetch["complete"]})


def load_file_titles():
    titles = []
    try:
        with open("utils/files.txt", "r") as file:
            for line in file:
                print(line)
                key, _ = line.strip().split(":")
                titles.append(key)
    except FileNotFoundError:
        pass
    return titles

def load_prompts(file_path):
    """
    Load prompts from a text file and return as a dictionary.
    """
    prompts = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    prompts[key.strip()] = value.strip()
    return prompts

@app.route('/fetching_results', methods=['GET'])
def fetching_results():
    """
    Render fetching_results.html and start querying the database for predefined questions.
    """
    return render_template('fetching_results.html')


# prepopulated_questions = {
#     "Net Sales": "What is the Net Sales?",
#     "Gross Profit": "What is the Gross Profit?",
#     "Debt/Equity Ratio": "What is the Debt/Equity Ratio?",
#     "Company Name": "What is the name of the company mentioned in the document?"
# }

fetched_results = {}  # Store results globally for simplicity


def query_vector_db(question, db, model, latest_file):
    answer = query_rag_latest(question, db, model, latest_file)  
    return answer


def redirect_to_fetching_results():
    # Wait until the database population is complete
    while not processing_status_upload["complete"]:
        pass

    # Once population is complete, move to querying the database
    processing_status_fetch["complete"] = False
    threading.Thread(target=run_query_database).start()


# def run_query_database():
#     """
#     Query the database for prepopulated questions and update the fetched_results.
#     """
#     global fetched_results
#     global processing_status_fetch

#     # Simulate querying the database
#     results = {}
#     for key, question in prepopulated_questions.items():
#         results[key] = query_vector_db(question, db, model)

#     fetched_results.update(results)

#     # Mark fetching as complete and redirect to analyze page
#     processing_status_fetch["complete"] = True
#     with app.app_context():
#         return render_template('analyze.html', data=fetched_results)

def run_query_database():
    """
    Query the database for prepopulated questions and update the fetched_results.
    """
    global fetched_results
    global processing_status_fetch

    # Get all items in the database
    all_items = db.get(include=["metadatas", "documents"])
    latest_file = None
    latest_file_ids = []

    # Extract the latest file name from metadata
    if all_items and "metadatas" in all_items and all_items["metadatas"]:
        latest_metadata = all_items["metadatas"][-1]  # Get the last metadata item
        latest_file = latest_metadata.get("source")

    # Collect IDs for the latest file
    if latest_file:
        latest_file_ids = [
            idx for idx, metadata in enumerate(all_items["metadatas"])
            if metadata.get("source") == latest_file
        ]

    # Save the latest file and IDs globally
    global latest_file_data
    latest_file_data = {"filename": latest_file, "ids": latest_file_ids}

    # Simulate querying the database for each prepopulated question
    results = {}
    prepopulated_questions = load_prompts(PROMPTS_FILE_PATH)
    print(prepopulated_questions)
    for key, question in prepopulated_questions.items():
        results[key] = query_vector_db(question, db, model, latest_file)

    fetched_results.update(results)

    # Mark fetching as complete and redirect to analyze page
    processing_status_fetch["complete"] = True
    with app.app_context():
        return render_template('analyze.html', data=fetched_results)



@app.route('/analyze', methods=['GET'])
def analyze():
    """
    Render analyze.html with fetched results.
    """
    return render_template('analyze.html', data=fetched_results)


if __name__ == '__main__':
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
