import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title("AskSherlock AI Book Assistant 🏦")

# Load small portion of data (important for cloud)
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv", on_bad_lines='skip')
    df = df.head(500)  # LIMIT DATASET 
    df["combined_text"] = df["title"] + " by " + df["authors"]
    return df

df = load_data()

query = st.text_input("Ask for a book recommendation")

if query:
    with st.spinner("Thinking... "):

        # Use smaller model (VERY IMPORTANT)
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

        embeddings = model.encode(df["combined_text"].tolist())

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        query_vector = model.encode([query])
        distances, indices = index.search(np.array(query_vector), 5)

        results = df.iloc[indices[0]]

    st.write("### Recommended Books")

    for _, row in results.iterrows():
        st.write(f"**{row['title']}** by {row['authors']}")
        st.write(f"Rating: {row['average_rating']}")
        st.write("---")