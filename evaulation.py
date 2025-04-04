#Sample evaluation dataset

evaluation_data = [
    {"question": "Which month has most bookings?", "ground_truth": "October"},
    {"question": "Show me the total revenue for July 2017?", "ground_truth": "$247,020"},
    {"question": "Which location has the highest booking cancellation?", "ground_truth": "City Hotel"},
    {"question": "Which country has the least booking?", "ground_truth": "NOR"},
]

# defining the evaluation function

!pip install sklearn
from sklearn.metrics import f1_score

def evaluate_rag_system(evaluation_data, k=5):
    total_questions = len(evaluation_data)
    correct_answers = 0
    all_generated_answers = []
    all_ground_truths = []

    for item in evaluation_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Get the answer from the RAG system
        generated_answer = answer_query_with_rag(question, k)

        # Store generated and ground truth answers for evaluation
        all_generated_answers.append(generated_answer)
        all_ground_truths.append(ground_truth)

        # Check if the generated answer matches the ground truth
        if generated_answer.strip().lower() == ground_truth.strip().lower():
            correct_answers += 1

    # Calculate Exact Match (EM) and F1 Score
    accuracy = correct_answers / total_questions
    f1 = f1_score(all_ground_truths, all_generated_answers, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Run the evaluation
evaluate_rag_system(evaluation_data)