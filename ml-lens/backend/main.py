from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="ML Lens API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StatItem(BaseModel):
    label: str
    value: str
    subtext: str

class Evaluation(BaseModel):
    id: int
    name: str
    status: str
    score: Optional[float] = None
    created_at: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/stats", response_model=List[StatItem])
async def get_stats():
    return [
        {"label": "Total Evaluations", "value": "24", "subtext": "This month"},
        {"label": "Pass Rate", "value": "92%", "subtext": "Baseline models"},
        {"label": "Avg Score", "value": "8.4/10", "subtext": "All evaluations"}
    ]

@app.get("/api/evaluations", response_model=List[Evaluation])
async def get_evaluations():
    return [
        {"id": 1, "name": "GPT-4 vs Claude", "status": "completed", "score": 8.7, "created_at": "2024-01-15"},
        {"id": 2, "name": "Summarization Task", "status": "in-progress", "score": None, "created_at": "2024-01-16"},
        {"id": 3, "name": "Code Generation", "status": "completed", "score": 8.2, "created_at": "2024-01-14"}
    ]

@app.post("/api/evaluations", response_model=Evaluation)
async def create_evaluation(eval: Evaluation):
    return {"id": 4, **eval.dict(), "created_at": datetime.now().isoformat()}
