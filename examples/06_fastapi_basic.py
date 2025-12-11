"""
06_fastapi_basic.py — FastAPI Integration (Simple)
=================================================

This example shows the simplest way to use pyaiagent with FastAPI.
Perfect for prototypes and simple APIs.

What you'll learn:
  • Creating a FastAPI app with a pyaiagent agent
  • Reusing the same agent instance for all requests
  • A simple POST endpoint for chat

Key concept:
  Create the agent ONCE at module level and reuse it for all requests.
  Creating a new agent per request is wasteful and slow!

Prerequisites:
  pip install pyaiagent fastapi uvicorn
  export OPENAI_API_KEY="sk-..."

Run this example:
  uvicorn examples.06_fastapi_basic:app --reload

Test it:
  curl -X POST "http://localhost:8000/chat" \\
       -H "Content-Type: application/json" \\
       -d '{"message": "Hello!"}'

Or visit: http://localhost:8000/docs for interactive docs.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pyaiagent import OpenAIAgent, OpenAIAgentProcessError


# =============================================================================
# Define the Agent
# =============================================================================

class ChatAssistant(OpenAIAgent):
    """
    You are a helpful, friendly assistant for a chat API.
    Keep responses concise but informative.
    Be professional and courteous.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.5
        max_output_tokens = 1024


# =============================================================================
# Create the Agent Instance (REUSED for all requests)
# =============================================================================
#
# IMPORTANT: Create the agent ONCE at module level.
# This is much more efficient than creating a new agent per request!

agent = ChatAssistant()


# =============================================================================
# Create FastAPI App
# =============================================================================

app = FastAPI(
    title="Chat API",
    description="A simple chat API powered by pyaiagent",
    version="1.0.0"
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str
    session_id: str | None = None  # Optional: for tracking purposes


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str
    session_id: str
    tokens_used: int


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the AI assistant.

    - **message**: Your message to the assistant
    - **session_id**: Optional session identifier (for tracking, not memory)

    Note: This basic example does NOT maintain conversation memory.
    See 07_fastapi_with_lifespan.py for multi-turn conversations.
    """
    try:
        result = await agent.process(
            input=request.message,
            session=request.session_id
        )

        return ChatResponse(
            response=result["output"],
            session_id=result["session"],
            tokens_used=result["tokens"]["total_tokens"]
        )

    except OpenAIAgentProcessError as e:
        # Handle all pyaiagent errors
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for load balancers."""
    return {"status": "healthy", "agent": "ChatAssistant"}


@app.get("/")
async def root() -> dict:
    """Welcome message with usage instructions."""
    return {
        "message": "Welcome to the Chat API!",
        "usage": {
            "endpoint": "POST /chat",
            "body": {
                "message": "Your message here",
                "session_id": "optional-tracking-id"
            },
            "example_curl": (
                'curl -X POST "http://localhost:8000/chat" '
                '-H "Content-Type: application/json" '
                "-d '{\"message\": \"Hello!\"}'"
            ),
        },
        "docs": "Visit /docs for interactive API documentation"
    }


# =============================================================================
# Run Directly (Development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("🚀 Starting Chat API Server")
    print("=" * 50)
    print("📖 Interactive docs: http://localhost:8000/docs")
    print("🔗 Health check:     http://localhost:8000/health")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)


# =============================================================================
# Limitations of This Basic Example
# =============================================================================
#
# 1. NO CONVERSATION MEMORY
#    Each request is independent. The agent doesn't remember previous
#    messages. See 07_fastapi_with_lifespan.py for memory support.
#
# 2. NO GRACEFUL SHUTDOWN
#    The OpenAI client isn't properly closed on shutdown.
#    See 07_fastapi_with_lifespan.py for proper lifecycle management.
#
# 3. BASIC ERROR HANDLING
#    All errors return 500. In production, you'd want different
#    status codes for different error types. See 08_error_handling.py.
#
# For production, always use the lifespan pattern in 07_fastapi_with_lifespan.py!

