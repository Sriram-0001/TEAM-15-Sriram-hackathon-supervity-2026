# core/api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.rag import ask

app = FastAPI(
    title="Telco RAG Support Assistant",
    version="0.1.0"
)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request Schema ----------
class AskRequest(BaseModel):
    query: str

# ---------- API Endpoint ----------
@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.query)

# ---------- Serve Frontend (MOUNT LAST) ----------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
