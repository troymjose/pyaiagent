"""
07_fastapi_with_lifespan.py — Production FastAPI Integration
============================================================

This is the RECOMMENDED way to integrate pyaiagent with FastAPI.
It includes proper lifecycle management, conversation memory,
and graceful shutdown.

What you'll learn:
  • FastAPI lifespan pattern for startup/shutdown
  • Storing agents in app.state
  • Conversation memory with session tracking
  • Proper cleanup with shutdown()

Key concepts:
  • Use lifespan to create agents on startup and clean up on shutdown
  • Store conversation history per session (Redis/DB in production)
  • Call shutdown() to close the shared OpenAI client

Prerequisites:
  pip install pyaiagent fastapi uvicorn
  export OPENAI_API_KEY="sk-..."

Run this example:
  uvicorn examples.07_fastapi_with_lifespan:app --reload

Test it:
  # First message
  curl -X POST "http://localhost:8000/chat" \\
       -H "Content-Type: application/json" \\
       -d '{"message": "Hi! My name is Alex.", "session_id": "user-123"}'

  # Second message (agent remembers!)
  curl -X POST "http://localhost:8000/chat" \\
       -H "Content-Type: application/json" \\
       -d '{"message": "What is my name?", "session_id": "user-123"}'
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from pyaiagent import OpenAIAgent, OpenAIAgentProcessError, shutdown


# =============================================================================
# Define the Agent
# =============================================================================

class ConversationalAssistant(OpenAIAgent):
    """
    You are a helpful conversational assistant.
    You remember everything the user tells you within the same session.
    Be friendly, helpful, and concise.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.5
        max_output_tokens = 1024


# =============================================================================
# Session Storage (In-Memory for Demo)
# =============================================================================
#
# ⚠️  In production, use Redis, PostgreSQL, or another persistent store!
#     This in-memory dict will be lost on server restart.

session_storage: dict[str, list[dict[str, Any]]] = {}


# =============================================================================
# Lifespan Context Manager
# =============================================================================
#
# This is the recommended pattern for managing application lifecycle.
# - Code BEFORE yield runs on startup
# - Code AFTER yield runs on shutdown

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle.

    Startup:  Create the agent and store in app.state
    Shutdown: Clean up the OpenAI client (prevents connection leaks)
    """
    # ─── STARTUP ───────────────────────────────────────────────────────────
    print("🚀 Starting up...")
    app.state.agent = ConversationalAssistant()
    print("✅ Agent created and ready!")

    yield  # ← Application runs here

    # ─── SHUTDOWN ──────────────────────────────────────────────────────────
    print("🛑 Shutting down...")

    # Close the shared OpenAI client for this event loop.
    # This is IMPORTANT to prevent connection leaks!
    await shutdown()

    print("✅ Cleanup complete!")


# =============================================================================
# Create FastAPI App with Lifespan
# =============================================================================

app = FastAPI(
    title="Conversational Chat API",
    description="A production-ready chat API with conversation memory",
    version="1.0.0",
    lifespan=lifespan  # ← Attach the lifespan manager
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str
    session_id: str  # Required for conversation tracking

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello! My name is Alex.",
                "session_id": "user-123"
            }
        }


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str
    session_id: str
    turn_count: int
    tokens_used: int


class SessionInfo(BaseModel):
    """Information about a chat session."""
    session_id: str
    message_count: int
    exists: bool


# =============================================================================
# Chat Endpoint (with Memory!)
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Send a message and get a response with conversation memory.

    The agent remembers previous messages within the same session_id.

    - **message**: Your message to the assistant
    - **session_id**: Unique identifier for this conversation

    Example conversation:
    1. "Hi! My name is Alex." → Agent greets Alex
    2. "What is my name?" → "Your name is Alex!"
    """
    try:
        # Get the agent from app.state (created during startup)
        agent: ConversationalAssistant = req.app.state.agent

        # Retrieve existing conversation history (or empty list)
        history = session_storage.get(request.session_id, [])

        # Process the message WITH conversation history
        result = await agent.process(
            input=request.message,
            session=request.session_id,
            history=history  # ← This enables memory!
        )

        # Save the UPDATED conversation history
        session_storage[request.session_id] = result["history"]

        # Count turns (user messages)
        turn_count = sum(
            1 for msg in result["history"]
            if isinstance(msg, dict) and msg.get("role") == "user"
        )

        return ChatResponse(
            response=result["output"],
            session_id=result["session"],
            turn_count=turn_count,
            tokens_used=result["tokens"]["total_tokens"]
        )

    except OpenAIAgentProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# =============================================================================
# Session Management Endpoints
# =============================================================================

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Get information about a specific session."""
    messages = session_storage.get(session_id, [])
    return SessionInfo(
        session_id=session_id,
        message_count=len(messages),
        exists=session_id in session_storage
    )


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> dict:
    """
    Clear conversation history for a session.
    Use this to start a fresh conversation.
    """
    if session_id in session_storage:
        del session_storage[session_id]
        return {"message": f"Session '{session_id}' cleared", "success": True}
    return {"message": f"Session '{session_id}' not found", "success": True}


@app.get("/sessions")
async def list_sessions() -> dict:
    """List all active sessions (with message counts)."""
    sessions = {
        sid: len(messages)
        for sid, messages in session_storage.items()
    }
    return {
        "sessions": sessions,
        "total_sessions": len(sessions)
    }


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health")
async def health_check() -> dict:
    """Health check for load balancers and monitoring."""
    return {
        "status": "healthy",
        "active_sessions": len(session_storage)
    }


@app.get("/")
async def root() -> dict:
    """API information and usage guide."""
    return {
        "name": "Conversational Chat API",
        "version": "1.0.0",
        "features": [
            "Multi-turn conversation memory",
            "Session management",
            "Proper lifecycle handling",
            "Graceful shutdown"
        ],
        "endpoints": {
            "POST /chat": "Send a message (requires session_id)",
            "GET /sessions": "List all active sessions",
            "GET /sessions/{id}": "Get session info",
            "DELETE /sessions/{id}": "Clear a session"
        },
        "example": {
            "first_message": 'POST /chat {"message": "My name is Alex", "session_id": "123"}',
            "second_message": 'POST /chat {"message": "What is my name?", "session_id": "123"}'
        },
        "docs": "/docs"
    }


# =============================================================================
# Run Directly
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 Starting Production Chat API with Lifespan")
    print("=" * 60)
    print("📖 Interactive docs: http://localhost:8000/docs")
    print("🔗 Health check:     http://localhost:8000/health")
    print("💬 Sessions:         http://localhost:8000/sessions")
    print("=" * 60)
    print("\n💡 Try this conversation flow:")
    print('   1. POST /chat {"message": "Hi! I am Alex", "session_id": "demo"}')
    print('   2. POST /chat {"message": "What is my name?", "session_id": "demo"}')
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)


# =============================================================================
# Production Considerations
# =============================================================================
#
# 1. SESSION STORAGE
#    Replace the in-memory dict with Redis or a database:
#
#    import redis.asyncio as redis
#
#    @asynccontextmanager
#    async def lifespan(app: FastAPI):
#        app.state.redis = await redis.from_url("redis://localhost")
#        app.state.agent = ConversationalAssistant()
#        yield
#        await app.state.redis.close()
#        await shutdown()
#
# 2. SESSION EXPIRY
#    Implement TTL for sessions to clean up abandoned conversations:
#
#    await redis.setex(f"session:{session_id}", 3600, json.dumps(messages))
#
# 3. MESSAGE LIMITS
#    Truncate long conversations to control token usage:
#
#    MAX_MESSAGES = 20
#    if len(history) > MAX_MESSAGES:
#        history = history[-MAX_MESSAGES:]
#
# 4. RATE LIMITING
#    Add rate limiting to prevent abuse:
#
#    from slowapi import Limiter
#    limiter = Limiter(key_func=get_remote_address)
#
# 5. AUTHENTICATION
#    Secure endpoints with JWT or API keys:
#
#    from fastapi.security import APIKeyHeader
#    api_key_header = APIKeyHeader(name="X-API-Key")

