"""
Hybrid + Agentic + Confidence-based RAG for Telecom Support
----------------------------------------------------------
Improvements:
- Robust query rewriting
- Softer confidence gating
- Clear Gemini health check
- Better debugging
- Reduced false rejections
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai


from core.embeddings import load_faiss_index, load_documents


# ================= CONFIG =================
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

REJECT_THRESHOLD = 0.25      # softer
ESCALATE_THRESHOLD = 0.55    # softer

MAX_MEMORY = 3


TELECOM_KEYWORDS = {
    "billing": ["bill", "payment", "charge", "invoice"],
    "network": ["network", "signal", "speed", "slow", "4g", "5g", "lte", "coverage"],
    "sim": ["sim", "activation", "swap", "port"],
    "plan": ["plan", "recharge", "data", "pack"]
}


# ================= INIT =================
embedder = SentenceTransformer(EMBEDDING_MODEL)

index = load_faiss_index()
documents = load_documents()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))



SESSION_MEMORY = []


# ================= UTILITIES =================
def detect_domain(text: str) -> str:
    text = text.lower()
    for domain, keywords in TELECOM_KEYWORDS.items():
        if any(k in text for k in keywords):
            return domain
    return "general"


def rewrite_query(query: str) -> str:
    """
    Agentic query rewriting (keyword-based, not exact match)
    """
    q = query.lower()

    if "slow" in q and ("network" in q or "internet" in q):
        return "low mobile data speed issue"

    if "no signal" in q or ("signal" in q and "not" in q):
        return "mobile network signal not available"

    if "sim" in q and ("not working" in q or "issue" in q):
        return "sim activation or sim failure issue"

    return query


def compute_confidence(distances, retrieved_docs, query):
    # FAISS similarity score
    best_distance = float(distances[0])
    vector_score = max(0.0, 1.0 - (best_distance / 2.0))

    # Intent score (soft)
    domain = detect_domain(query)
    intent_score = 0.5 if domain != "general" else 0.0

    # Keyword overlap score
    keyword_hits = 0
    for doc in retrieved_docs:
        for group in TELECOM_KEYWORDS.values():
            if any(k in doc.lower() for k in group):
                keyword_hits += 1
                break

    keyword_score = keyword_hits / max(1, len(retrieved_docs))

    confidence = (
        0.6 * vector_score +
        0.25 * intent_score +
        0.15 * keyword_score
    )

    return round(confidence, 2)


def update_memory(query, answer):
    SESSION_MEMORY.append((query, answer))
    if len(SESSION_MEMORY) > MAX_MEMORY:
        SESSION_MEMORY.pop(0)


def build_memory_context():
    if not SESSION_MEMORY:
        return ""
    return "\n".join(
        f"User: {q}\nAssistant: {a}"
        for q, a in SESSION_MEMORY
    )


# ================= MAIN RAG =================
def ask(query: str):
    rewritten_query = rewrite_query(query)
    domain = detect_domain(rewritten_query)

    # Embed + retrieve
    query_embedding = embedder.encode([rewritten_query]).astype("float32")
    distances, indices = index.search(query_embedding, TOP_K)

    if len(indices[0]) == 0:
        return reject("No relevant telecom data found")

    retrieved_docs = [documents[i]["text"] for i in indices[0]]
    source_ids = [documents[i]["source_id"] for i in indices[0]]

    confidence = compute_confidence(distances[0], retrieved_docs, rewritten_query)

    print(f"[DEBUG] Query='{query}' | Rewritten='{rewritten_query}'")
    print(f"[DEBUG] Domain={domain} | Confidence={confidence}")

    # ❌ HARD REJECT (clearly non-telecom)
    if confidence < REJECT_THRESHOLD and domain == "general":
        return {
            "answer": (
                "I’m designed to help with telecom-related issues like "
                "network, SIM, billing, or plans."
            ),
            "confidence": confidence,
            "decision": "rejected",
            "sources": []
        }

    # ⚠️ ESCALATE (low confidence telecom)
    if confidence < ESCALATE_THRESHOLD:
        return {
            "answer": (
                f"This seems related to **{domain}**, but I’m not fully confident. "
                "It may require human assistance. Would you like me to escalate?"
            ),
            "confidence": confidence,
            "decision": "escalate",
            "escalation_team": domain,
            "sources": list(set(source_ids))
        }

    # ✅ HIGH CONFIDENCE → GEMINI
    memory_context = build_memory_context()
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a telecom customer support assistant.

Conversation History:
{memory_context}

Knowledge Base:
{context}

User Question:
{rewritten_query}

Respond with:
1. Issue Summary
2. Likely Cause (based on similar tickets)
3. Step-by-step Resolution
4. Escalation Note (if unresolved)
"""
    try:
        print("[DEBUG] 🔥 Calling Gemini")
        response = client.models.generate_content(
            model="gemini-pro-latest",
            contents=prompt
        )
        answer = response.text.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"



    update_memory(query, answer)

    return {
        "answer": answer,
        "confidence": confidence,
        "decision": "answered",
        "domain": domain,
        "sources": list(set(source_ids))
    }


def reject(reason):
    return {
        "answer": f"I can’t assist with this request. ({reason})",
        "confidence": 0.0,
        "decision": "rejected",
        "sources": []
    }
