# InnSight

# Hotel Analytics and RGA System (Google Colab)

This repository contains a Google Colab notebook for hotel booking analytics and a Retrieval Augmented Generation (RAG) system.


### **Installation Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/prajaktabommat21005/InnSight
    cd Hotel_Analytics_RGA
    ```

2.  **Open the Colab Notebook:**

    * Open `Hotel_Analytics_RGA.ipynb` in Google Colab.

3.  **Data:**

    * The original dataset (`hotel_bookings.csv`) is in the `data/` directory.
    * Colab will create `cleaned_hotel_bookings.csv` and `analytics.json` in the `data/` directory during execution.

4.  **Run the Notebook:**

    * Execute the cells in the Colab notebook sequentially. The notebook contains all the code for data processing, analytics, and LLM integration.

5.  **Dependencies:**

    * The Colab notebook includes `!pip install` commands to install the necessary libraries. These include:
        * pandas
        * matplotlib
        * seaborn
        * scikit-learn
        * transformers
        * sentence-transformers
        * faiss-cpu.

## Project Structure

* `README.md`: This file.
* `codebase/`: Python modules.
    * `data_processing.py`: Data cleaning code.
    * `analytics.py`: Analytics code.
    * `llm_integration.py`: RAG code.
* `Hotel_Analytics_RGA.ipynb`: The main Colab notebook.
*`Sample queries.txt` consists of text cases along with the expected answers for evaluation.
* `report.md`: Implementation details.

