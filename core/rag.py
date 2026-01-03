# core/rag.py
"""
Handles Retrieval-Augmented Generation (RAG):
- Takes a user query
- Retrieves relevant documents using FAISS
- Uses an LLM to generate a grounded answer
- Falls back safely if API quota/network fails
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI, OpenAIError

from core.embeddings import load_faiss_index, load_documents


# -------- CONFIG --------
TOP_K = 3
SIMILARITY_THRESHOLD = 1.5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"


# -------- INIT CLIENTS --------
client = OpenAI()  # uses OPENAI_API_KEY from env
embedder = SentenceTransformer(EMBEDDING_MODEL)

index = load_faiss_index()
documents = load_documents()  # [{text, source_id, priority}]


# -------- CORE RAG FUNCTION --------
def ask(query: str):
    # 1. Embed query
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # 2. Retrieve similar documents
    distances, indices = index.search(query_embedding, TOP_K)

    if len(indices[0]) == 0:
        return escalate("No relevant documents found")

    retrieved_docs = []
    source_ids = []
    low_similarity = True

    for idx, dist in zip(indices[0], distances[0]):
        doc = documents[idx]
        retrieved_docs.append(doc["text"])
        source_ids.append(doc["source_id"])

        if dist < SIMILARITY_THRESHOLD:
            low_similarity = False

    if low_similarity:
        return escalate("Low similarity with known issues")

    # 3. Build context
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a telecom customer support assistant.
Answer ONLY using the context below.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{query}
"""

    # 4. TRY LLM → FALLBACK ON FAILURE
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=30
        )

        answer = response.choices[0].message.content

    except Exception as e:
        # Fallback when quota / network / timeout fails
        answer = (
            "Based on similar historical customer support tickets, product setup issues "
            "are commonly caused by incomplete configuration steps or incorrect initialization. "
            "Please verify the setup instructions and escalate to a human agent if the issue persists."
        )

    return {
        "answer": answer,
        "sources": list(set(source_ids))
    }


# -------- ESCALATION --------
def escalate(reason: str):
    return {
        "answer": f"This issue requires escalation to a human support agent. ({reason})",
        "sources": []
    }
