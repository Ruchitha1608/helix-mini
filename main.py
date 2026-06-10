from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.voice import router as voice_router
from app.api.session import router as session_router
from app.core.config import settings
import logging

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Helix Mini",
    description="Patient adherence voice AI — inspired by Synthio's Helix product",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router, prefix="/voice", tags=["voice"])
app.include_router(session_router, prefix="/session", tags=["session"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "helix-mini"}

@app.get("/")
async def root():
    return {
        "service": "Helix Mini",
        "description": "Patient adherence voice AI with FDA-grounded compliance guardrails",
        "endpoints": {
            "websocket": "/voice/ws/{session_id}",
            "session_summary": "/session/{session_id}/summary",
            "health": "/health"
        }
    }
