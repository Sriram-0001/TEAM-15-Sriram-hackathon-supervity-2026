# core/embeddings.py
"""
Loads customer support datasets, converts them into embeddings,
and builds a FAISS index for retrieval.
"""

import os
import pickle
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------
DATA_DIR = "data"
INDEX_PATH = "data/faiss.index"
DOCS_PATH = "data/documents.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # words


# ---------------- OPTIONAL (UNUSED, SAFE TO KEEP) ----------------
def load_datasets():
    datasets = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(path)
            datasets.append(df)

    return datasets


# ---------------- TEXT EXTRACTION (EXPLICIT & CORRECT) ----------------
def extract_documents():
    documents = []

    # -------- Dataset 1: Telecom Agent–Customer Interaction --------
    df1 = pd.read_csv("data/CustomerInteractionData.csv")

    for idx, text in df1["CustomerInteractionRawText"].dropna().items():
        documents.append({
            "text": str(text),
            "source_id": f"interaction_{idx}",
            "priority": "Normal"
        })

    # -------- Dataset 2: Customer Support Tickets --------
    df2 = pd.read_csv("data/customer_support_tickets.csv")

    for _, row in df2.iterrows():
        text = row["Ticket Description"]
        if pd.notna(text):
            documents.append({
                "text": str(text),
                "source_id": f"ticket_{row['Ticket ID']}",
                "priority": row["Ticket Priority"]
            })

    if len(documents) == 0:
        raise ValueError("❌ No documents extracted from datasets")

    print(f"✅ Extracted {len(documents)} documents")
    return documents


# ---------------- CHUNKING ----------------
def chunk_text(text):
    words = text.split()
    chunks = []

    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)

    return chunks


# ---------------- BUILD FAISS INDEX ----------------
def build_faiss_index():
    raw_docs = extract_documents()

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    documents = []

    for doc in raw_docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            documents.append({
                "text": chunk,
                "source_id": doc["source_id"],
                "priority": doc["priority"]
            })

    if len(texts) == 0:
        raise ValueError("❌ No text chunks available for embedding")

    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and documents
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"✅ Built FAISS index with {len(documents)} chunks")


# ---------------- LOADERS FOR RAG ----------------
def load_faiss_index():
    return faiss.read_index(INDEX_PATH)


def load_documents():
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)
