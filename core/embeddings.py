"""
Builds FAISS index from telecom datasets
"""

import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DATA_DIR = "data"
INDEX_PATH = "data/faiss.index"
DOCS_PATH = "data/documents.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # words


# ---------- EXTRACT DOCUMENTS ----------
def extract_documents():
    documents = []

    # Dataset 1: Agent–Customer interactions
    df1 = pd.read_csv("data/CustomerInteractionData.csv")
    for idx, text in df1["CustomerInteractionRawText"].dropna().items():
        documents.append({
            "text": str(text),
            "source_id": f"interaction_{idx}",
            "priority": "Normal"
        })

    # Dataset 2: Support tickets
    df2 = pd.read_csv("data/customer_support_tickets.csv")
    for _, row in df2.iterrows():
        if pd.notna(row["Ticket Description"]):
            documents.append({
                "text": str(row["Ticket Description"]),
                "source_id": f"ticket_{row['Ticket ID']}",
                "priority": row["Ticket Priority"]
            })

    if not documents:
        raise ValueError("No documents extracted")

    print(f"✅ Extracted {len(documents)} documents")
    return documents


# ---------- CHUNKING ----------
def chunk_text(text):
    words = text.split()
    return [
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0, len(words), CHUNK_SIZE)
    ]


# ---------- BUILD INDEX ----------
def build_faiss_index():
    raw_docs = extract_documents()
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts, docs = [], []

    for doc in raw_docs:
        for chunk in chunk_text(doc["text"]):
            texts.append(chunk)
            docs.append({
                "text": chunk,
                "source_id": doc["source_id"],
                "priority": doc["priority"]
            })

    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ FAISS index built with {len(docs)} chunks")


# ---------- LOADERS ----------
def load_faiss_index():
    return faiss.read_index(INDEX_PATH)


def load_documents():
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)
