AskSherlock 

AskSherlock is a simple AI-based book recommendation system that suggests books based on the meaning of a user's query.
Instead of matching keywords, it uses embeddings and vector search to find similar books.

 Tech Stack

* Python
* SentenceTransformers
* FAISS
* Pandas / NumPy
* Streamlit

 Dataset

This project uses the **Goodbooks-10k dataset** from Kaggle:
https://www.kaggle.com/datasets/zygmunt/goodbooks-10k

Download `books.csv` and place it in the project folder before running the app.

Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

 Example Queries

* dark fantasy magic
* romantic historical fiction
* space sci fi adventure

Author: Valentina
