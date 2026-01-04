from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.rag import ask

app = FastAPI(
    title="Telco RAG Support Assistant",
    version="1.0.0"
)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SCHEMA ----------
class AskRequest(BaseModel):
    query: str


# ---------- API ----------
@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.query)


# ---------- FRONTEND ----------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
