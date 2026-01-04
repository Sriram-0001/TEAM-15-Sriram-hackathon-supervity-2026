"""
Hybrid + Confidence-based RAG for Telecom Support
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from core.embeddings import load_faiss_index, load_documents

# ---------- CONFIG ----------
TOP_K = 5
SIMILARITY_THRESHOLD = 1.5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Confidence thresholds
REJECT_THRESHOLD = 0.35
ESCALATE_THRESHOLD = 0.65

TELECOM_KEYWORDS = [
    "sim", "network", "billing", "recharge",
    "plan", "activation", "signal", "data"
]

# ---------- INIT ----------
embedder = SentenceTransformer(EMBEDDING_MODEL)
index = load_faiss_index()
documents = load_documents()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-pro")


# ---------- CONFIDENCE ----------
def compute_confidence(distances, retrieved_docs, query):
    avg_distance = float(np.mean(distances))
    vector_score = max(0, 1 - (avg_distance / 2))

    keyword_hits = sum(
        any(k in doc.lower() for k in TELECOM_KEYWORDS)
        for doc in retrieved_docs
    )
    keyword_score = keyword_hits / max(1, len(retrieved_docs))

    intent_score = 1.0 if any(k in query.lower() for k in TELECOM_KEYWORDS) else 0.0

    return round(
        0.5 * vector_score +
        0.3 * keyword_score +
        0.2 * intent_score,
        2
    )


# ---------- RAG ----------
def ask(query: str):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, TOP_K)

    if len(indices[0]) == 0:
        return reject("No relevant telecom information found")

    retrieved_docs = [documents[i]["text"] for i in indices[0]]
    source_ids = [documents[i]["source_id"] for i in indices[0]]

    confidence = compute_confidence(distances[0], retrieved_docs, query)

    # ❌ OUT OF SCOPE
    if confidence < REJECT_THRESHOLD:
        return {
            "answer": (
                "I’m designed to assist only with telecom-related queries. "
                "I can’t help with this request."
            ),
            "confidence": confidence,
            "sources": []
        }

    # ⚠️ ESCALATE
    if confidence < ESCALATE_THRESHOLD:
        return {
            "answer": (
                "This request may require human assistance. "
                "Would you like me to connect you to a support agent?"
            ),
            "confidence": confidence,
            "sources": list(set(source_ids))
        }

    # ✅ HIGH CONFIDENCE → LLM
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a telecom customer support assistant.

Give:
1. Issue summary
2. Reasoning (based on similar past tickets)
3. Clear step-by-step resolution
4. Escalation note if needed

Use ONLY the context below.

Context:
{context}

Question:
{query}
"""

    try:
        response = llm.generate_content(prompt)
        answer = response.text
    except Exception:
        answer = (
            "Based on similar telecom issues, this problem may relate to configuration "
            "or activation. Please verify setup steps or contact support if unresolved."
        )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": list(set(source_ids))
    }
