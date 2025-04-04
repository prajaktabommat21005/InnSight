from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

app = Flask(__name__)

# Load data, models, and index 

df1 = pd.read_csv("data/cleaned_hotel_bookings.csv")
df1['text'] = df1.apply(lambda row: f"Hotel: {row['hotel']}, Country: {row['country']}, Lead Time: {row['lead_time']}, Cancelled: {row['is_canceled']}, Arrival Date: {row['arrival_date_year']}-{row['arrival_date_month']}", axis=1)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = embedding_model.encode(df1['text'].tolist(), convert_to_numpy=True)
index = faiss.IndexFlatL2(text_embeddings.shape[1])
index.add(text_embeddings)
texts = df1['text'].tolist()
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
with open("data/analytics.json", "r") as f:
    analytics = json.load(f)

def answer_query_with_rag(question, k=5):
    question_embedding = embedding_model.encode([question])[0]
    _, indices = index.search(np.array([question_embedding]), k)
    retrieved_texts = [texts[i] for i in indices[0]]
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route("/query", methods=["POST"])
def query_hotel_data():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = answer_query_with_rag(question)
    return jsonify({"question": question, "answer": answer})

@app.route("/analytics", methods=["GET"])
def get_analytics():
    return jsonify(analytics)

if __name__ == '__main__':
    app.run(debug=True)