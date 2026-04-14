from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from app.agent import AgentOrchestrator

app = FastAPI()

# Global orchestrator and session management
orchestrator = AgentOrchestrator(channel="chat")
sessions = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    state: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.session_id or request.session_id not in sessions:
        ctx = orchestrator.new_session()
        sessions[ctx.session_id] = ctx
        print(f"DEBUG: Created NEW session {ctx.session_id}")
    else:
        ctx = sessions[request.session_id]
        print(f"DEBUG: Using EXISTING session {ctx.session_id} (turn {ctx.turn_number})")

    response_text = orchestrator.process_turn(ctx, text_input=request.message)

    return ChatResponse(
        session_id=ctx.session_id,
        response=response_text,
        state=ctx.state.value
    )

@app.post("/reset")
async def reset(session_id: Optional[str] = None):
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"status": "reset"}

# Serve the frontend static files — MUST be mounted after API routes
if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
