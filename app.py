import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
df = pd.read_csv("books.csv")

df["combined_text"] = df["title"] + " by " + df["authors"]

# Create embeddings
embeddings = model.encode(df["combined_text"].tolist())

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

st.title("AskSherlock AI Book Assistant")

query = st.text_input("Ask for a book recommendation")

if query:
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), 5)

    results = df.iloc[indices[0]]

    st.write("### Recommended Books")

    for _, row in results.iterrows():
        st.write(f"**{row['title']}** by {row['authors']}")
        st.write(f"Rating: {row['average_rating']}")
        st.write("---")