# core/api.py
"""
FastAPI wrapper for RAG pipeline.
Exposes /ask endpoint for frontend access.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from core.rag import ask

app = FastAPI(title="Telco RAG Support Assistant")


# -------- Request Schema --------
class AskRequest(BaseModel):
    query: str


# -------- Response Schema --------
class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# -------- API Endpoint --------
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    result = ask(req.query)
    return result
