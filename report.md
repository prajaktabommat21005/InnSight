# Implementation Report

## **Overview**
This report details the implementation of a system for analyzing hotel booking data and providing question-answering capabilities using Retrieval Augmented Generation (RAG). The system encompasses data preprocessing, exploratory data analysis, and integration with a Large Language Model (LLM).

## **Implementation Choices**
- **Data Cleaning:** Used pandas for data manipulation, handling missing values, and ensuring data consistency.
- **Analytics:** Calculated monthly revenue, cancellation rates, and other statistics to provide insights into booking trends.
- **RAG Integration:** Utilized Sentence Transformers for embedding and FAISS for efficient similarity search, combined with a T5 model for generating answers.

## **Challenges**
- **Data Quality:** The original dataset had missing values and inconsistencies that required thorough cleaning.
- **Model Performance:** Ensuring the RAG model provided accurate and relevant answers based on the embeddings was a key challenge.
- **Scalability:** As the dataset grows, maintaining performance in both analytics and query response times will be crucial.
- **API Integration:** The API integration phase has been particularly challenging. A significant contributing factor to this difficulty is the complexity of integrating the RAG system's logic into a web service framework. The need to carefully manage data flow, handle asynchronous operations, and optimize performance for real-time responsiveness has required a deeper understanding of both the RAG system and API development best practices. As a result, I am still actively working on refining the API to meet the project's requirements.
