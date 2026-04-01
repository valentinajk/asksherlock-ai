import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title("AskSherlock AI Book Assistant 🚀")

# -------------------------------
# Load dataset (light + safe)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv", on_bad_lines='skip')
    df = df.head(300)  # LIMIT DATASET (important for deployment)
    df["combined_text"] = df["title"] + " by " + df["authors"]
    return df

df = load_data()

# -------------------------------
# User input
# -------------------------------
query = st.text_input("Ask for a book recommendation")

# -------------------------------
# Recommendation logic
# -------------------------------
if query:
    with st.spinner("Thinking... 🤖 Please wait"):

        # Load smaller model (fast + low memory)
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

        # Create embeddings
        embeddings = model.encode(df["combined_text"].tolist())

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        # Search
        query_vector = model.encode([query])
        distances, indices = index.search(np.array(query_vector), 5)

        results = df.iloc[indices[0]]

    # -------------------------------
    # Display results
    # -------------------------------
    st.write("### 📚 Recommended Books")

    for _, row in results.iterrows():
        st.write(f"**{row['title']}** by {row['authors']}")
        st.write(f"⭐ Rating: {row['average_rating']}")
        st.write("---")
