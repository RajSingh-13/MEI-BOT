import os
import re
import json
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from collections import Counter, deque
import google.generativeai as genai
from datetime import datetime

# ====== CONFIG ======
UPLOAD_FOLDER = 'data'
LOG_FILE = os.path.join(UPLOAD_FOLDER, "query_log.json")
ALLOWED_EXTENSIONS = {'pdf'}
FREQ_FILE = os.path.join(UPLOAD_FOLDER, "frequent_answers.json")

# API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing.")

genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# Flask app
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
pdf_data = {}
pdf_chunks = []
pdf_embeddings = None

# Load frequent answers
if os.path.exists(FREQ_FILE):
    with open(FREQ_FILE, "r") as f:
        frequent_answers = json.load(f)
else:
    frequent_answers = {}

# Load existing logs
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        query_log = json.load(f)
else:
    query_log = []

# Maintain rolling chat log in memory (last 50)
chat_history = deque(maxlen=50)

conversation_state = {}

# Reference semantic domain
reference_texts = [
    "Mitsubishi Electric air conditioner troubleshooting",
    "Mitsubishi Electric AC installation manual",
    "AC error code list for Mitsubishi Electric",
    "Mitsubishi HVAC maintenance tips",
    "Mitsubishi Electric indoor outdoor unit repair",
    "Mitsubishi Electric warranty and service in India",
    "AC cooling problem diagnosis",
    "Air conditioner water leakage Mitsubishi",
    "How to fix Mitsubishi AC noise issue",
    "Reset Mitsubishi AC error code"
]
reference_embeddings = embedder.encode(reference_texts)
faiss.normalize_L2(np.array(reference_embeddings))

# ====== HELPERS ======
def log_query(query, score, relevance):
    """Log query with details"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "similarity": round(score, 3),
        "relevance": relevance
    }
    query_log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(query_log, f, indent=2)

def check_query_relevance(query):
    query_emb = embedder.encode([query])
    faiss.normalize_L2(query_emb)
    sims = np.dot(query_emb, np.array(reference_embeddings).T)
    max_sim = float(np.max(sims))
    threshold = 0.35
    is_relevant = max_sim >= threshold

    print(f"[DEBUG] Query: {query} | Score: {max_sim:.3f} | Relevant: {is_relevant}")
    log_query(query, max_sim, "Relevant" if is_relevant else "Low relevance")
    return is_relevant, max_sim

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except:
            pass
    chunks = re.split(r'\n\s*\n', text)
    return [c.strip() for c in chunks if c.strip()]

def update_pdf_index(filename, chunks):
    global pdf_chunks, pdf_embeddings
    pdf_data[filename] = chunks
    all_chunks = [c for chunk_list in pdf_data.values() for c in chunk_list]
    pdf_chunks = all_chunks
    embeddings = embedder.encode(all_chunks)
    faiss.normalize_L2(embeddings)
    pdf_embeddings = np.array(embeddings).astype('float32')
    index.reset()
    index.add(pdf_embeddings)

def search_pdf(query, top_k=3):
    if not pdf_chunks:
        return ["No PDF manual uploaded. Please upload one first."]
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(np.array(q_emb).astype('float32'), top_k)
    return [pdf_chunks[i] for i in I[0] if i < len(pdf_chunks)]

def ai_generate_answer(user_query, context_text):
    prompt = f"""
    You are a Mitsubishi Electric India support assistant.
    Extract only useful troubleshooting steps from the manual below.
    Focus only on solving the user's problem in clear, step-by-step language.

    User's problem: {user_query}

    Manual context:
    {context_text}

    Write a concise, clear, and actionable solution.
    """
    try:
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

def suggest_related_questions(user_message):
    keywords_map = {
        "error": ["Error code E7", "Reset error codes", "Common AC error issues"],
        "leakage": ["Water leakage from indoor unit", "Pipe blockage issue"],
        "maintenance": ["Maintenance tips", "Filter cleaning process", "Seasonal checkup steps"],
        "cooling": ["AC not cooling enough", "Improve cooling performance"],
        "power": ["AC not turning on", "Power supply troubleshooting"]
    }
    suggestions = []
    for key, sugg in keywords_map.items():
        if key in user_message.lower():
            suggestions.extend(sugg)
    return suggestions[:3] if suggestions else ["Maintenance tips", "Error code guide", "Cooling efficiency tips"]

# ====== ROUTES ======
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.remote_addr
    user_message = request.json.get('message', '').strip()

    # If awaiting feedback
    if conversation_state.get(user_id, {}).get("awaiting_feedback"):
        if user_message.lower() in ["yes", "y"]:
            q = conversation_state[user_id]["last_question"]
            a = conversation_state[user_id]["last_answer"]
            frequent_answers[q] = a
            with open(FREQ_FILE, "w") as f:
                json.dump(frequent_answers, f)
            conversation_state[user_id] = {}
            return jsonify({"reply": "Glad to hear that! Saved for future quick answers.", "suggestions": []})
        elif user_message.lower() in ["no", "n"]:
            last_q = conversation_state[user_id]["last_question"]
            pdf_matches = search_pdf(last_q, top_k=5)
            context = "\n\n".join(pdf_matches)
            detailed_answer = ai_generate_answer(last_q + " Please give more details", context)
            conversation_state[user_id] = {}
            return jsonify({"reply": detailed_answer, "suggestions": suggest_related_questions(last_q)})
        else:
            return jsonify({"reply": "Please reply Yes or No for better assistance.", "suggestions": []})

    # Semantic check
    is_relevant, similarity = check_query_relevance(user_message)

    # Frequent answers check
    if user_message in frequent_answers:
        return jsonify({"reply": frequent_answers[user_message], "suggestions": suggest_related_questions(user_message)})

    # Generate answer
    pdf_matches = search_pdf(user_message)
    context = "\n\n".join(pdf_matches)
    reply = ai_generate_answer(user_message, context)

    if not is_relevant:
        reply = "Note: This may be outside Mitsubishi Electric India scope.\n\n" + reply

    # Save for feedback
    conversation_state[user_id] = {"last_question": user_message, "last_answer": reply, "awaiting_feedback": True}
    chat_history.append({"user": user_message, "bot": reply})

    return jsonify({"reply": reply + "\n\nDid this solve your problem? (Yes / No)", 
                    "suggestions": suggest_related_questions(user_message)})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'status': 'error', 'message': 'No PDF file uploaded'})
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Only PDF files allowed'})

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)
    chunks = extract_text_from_pdf(filepath)
    update_pdf_index(file.filename, chunks)
    return jsonify({'status': 'success', 'message': f'PDF \"{file.filename}\" uploaded and indexed successfully'})

# ====== ADMIN DASHBOARD ======
@app.route('/admin')
def admin_dashboard():
    total_queries = len(query_log)
    relevance_counts = Counter(entry['relevance'] for entry in query_log)
    top_queries = Counter(entry['query'] for entry in query_log).most_common(5)
    top_answers = list(frequent_answers.items())[:5]
    return render_template('admin.html',
                           total_queries=total_queries,
                           relevance_counts=relevance_counts,
                           top_queries=top_queries,
                           top_answers=top_answers,
                           manuals=list(pdf_data.keys()),
                           logs=query_log[-20:])  # show last 20 logs

@app.route('/admin/delete_answer/<question>', methods=['POST'])
def delete_answer(question):
    if question in frequent_answers:
        del frequent_answers[question]
        with open(FREQ_FILE, "w") as f:
            json.dump(frequent_answers, f)
    return redirect(url_for('admin_dashboard'))

# ====== MAIN ======
if __name__ == "__main__":
    app.run(debug=True)
