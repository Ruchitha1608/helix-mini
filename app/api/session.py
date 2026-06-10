from fastapi import APIRouter, HTTPException
from app.services.session_store import session_store

router = APIRouter()


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get post-call structured summary for a session."""
    summary = session_store.get_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return summary


@router.get("/active")
async def list_active_sessions():
    """List all currently active sessions."""
    return {
        "active_sessions": list(session_store.sessions.keys()),
        "count": len(session_store.sessions)
    }
