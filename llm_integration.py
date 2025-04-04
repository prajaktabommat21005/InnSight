import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load cleaned data
df1 = pd.read_csv("data/cleaned_hotel_bookings.csv")

# Create text column for embedding
df1['text'] = df1.apply(lambda row: f"Hotel: {row['hotel']}, Country: {row['country']}, Lead Time: {row['lead_time']}, Cancelled: {row['is_canceled']}, Arrival Date: {row['arrival_date_year']}-{row['arrival_date_month']}", axis=1)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text into embeddings
text_embeddings = embedding_model.encode(df1['text'].tolist(), convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(text_embeddings.shape[1])
index.add(text_embeddings)

# Save text for retrieval
texts = df1['text'].tolist()

# Load FLAN-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def answer_query_with_rag(question, k=5):
    # Embed the user question
    question_embedding = embedding_model.encode([question])[0]

    # Retrieve top-k similar entries
    _, indices = index.search(np.array([question_embedding]), k)
    retrieved_texts = [texts[i] for i in indices[0]]

    # Create context prompt
    context = " ".join(retrieved_texts)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Tokenize and generate answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

# Ask user (for command-line interaction)
user_query = input("Ask a question about hotel bookings: ")
response = answer_query_with_rag(user_query)
print("\nAnswer:", response)