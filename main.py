import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title("AskSherlock AI Book Assistant 🏛")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv", on_bad_lines='skip')
    df = df.head(2000)  # you can increase if needed
    df["combined_text"] = df["title"] + " " + df["authors"]
    return df

df = load_data()

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# Create FAISS index (cached)
# -------------------------------
@st.cache_resource
def create_index(texts):
    model = load_model()
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

index = create_index(df["combined_text"].tolist())

# -------------------------------
# User input
# -------------------------------
query = st.text_input("Ask for a book recommendation")

# -------------------------------
# Search
# -------------------------------
if query:
    with st.spinner("Thinking... "):

        model = load_model()
        query_vector = model.encode([query])

        distances, indices = index.search(np.array(query_vector), 5)
        results = df.iloc[indices[0]]

    st.write("### 🏛 AI Recommendations")

    for _, row in results.iterrows():
        st.write(f"**{row['title']}** by {row['authors']}")
        st.write(f"⭐ Rating: {row['average_rating']}")
        st.write("---")