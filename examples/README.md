# pyaiagent Examples

Welcome to the pyaiagent examples! This directory contains production-quality examples that demonstrate every feature of the library.

## Getting Started

### Prerequisites

```bash
pip install pyaiagent
export OPENAI_API_KEY="sk-..."
```

### Running Examples

Each example is a standalone script:

```bash
# From the project root
python examples/00_basic_agent.py
```

For FastAPI examples:

```bash
pip install fastapi uvicorn
uvicorn examples.06_fastapi_basic:app --reload
```

---

## Example Overview

### Beginner Examples

| File | Description | Key Concepts |
|------|-------------|--------------|
| [`00_basic_agent.py`](00_basic_agent.py) | Your first agent in 10 lines | `OpenAIAgent`, docstrings, `process()` |
| [`01_conversation_memory.py`](01_conversation_memory.py) | Multi-turn conversations | `llm_messages`, chat loops |
| [`02_tools_basic.py`](02_tools_basic.py) | Give agents superpowers | `async`/`sync` tools, type hints, docstrings |

### Intermediate Examples

| File | Description | Key Concepts |
|------|-------------|--------------|
| [`03_structured_output.py`](03_structured_output.py) | Typed responses with Pydantic | `text_format`, `output_parsed` |
| [`04_config_advanced.py`](04_config_advanced.py) | All configuration options | `Config` class, temperature, timeouts |
| [`05_dynamic_instructions.py`](05_dynamic_instructions.py) | Runtime personalization | `instruction_params`, placeholders |

### Production Examples

| File | Description | Key Concepts |
|------|-------------|--------------|
| [`06_fastapi_basic.py`](06_fastapi_basic.py) | Simple FastAPI integration | REST API, agent reuse |
| [`07_fastapi_with_lifespan.py`](07_fastapi_with_lifespan.py) | Production FastAPI setup | Lifespan, `shutdown()`, sessions |
| [`08_error_handling.py`](08_error_handling.py) | Comprehensive error handling | Exceptions, logging, patterns |

### Advanced Examples

| File | Description | Key Concepts |
|------|-------------|--------------|
| [`09_inheritance_composition.py`](09_inheritance_composition.py) | Building agent hierarchies | Inheritance, tool reuse, routing |
| [`10_context_manager.py`](10_context_manager.py) | Async context managers | `async with`, cleanup, pipelines |
| [`11_message_formatting.py`](11_message_formatting.py) | Token optimization hooks | `format_llm_message`, `format_ui_message` |
| [`12_dependency_injection.py`](12_dependency_injection.py) | Injecting services via `__init__` | DB clients, API clients, testing |
| [`13_custom_client.py`](13_custom_client.py) | Custom OpenAI client configuration | `set_default_openai_client`, Azure, Ollama, proxies |

---

## Quick Reference

### Minimal Agent

```python
from pyaiagent import OpenAIAgent

class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""

agent = MyAgent()
result = await agent.process(input="Hello!")
print(result["output"])
```

### Agent with Tools

```python
class ToolAgent(OpenAIAgent):
    """You can use tools to help users."""

    # Async tool: for I/O-bound work (API calls, DB)
    async def get_weather(self, city: str) -> dict:
        """Get weather for a city."""
        return {"city": city, "temp": "22°C"}

    # Sync tool: for CPU-bound work (runs in thread pool)
    def calculate(self, expression: str) -> dict:
        """Evaluate a math expression."""
        return {"result": eval(expression)}
```

### Structured Output

```python
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

class StructuredAgent(OpenAIAgent):
    """Return structured responses."""

    class Config:
        text_format = Response
```

### Conversation Memory

```python
llm_messages = []
for user_input in messages:
    result = await agent.process(
        input=user_input,
        llm_messages=llm_messages
    )
    llm_messages = result["messages"]["llm"]
```

### Message Formatting (Token Optimization)

```python
class MyAgent(OpenAIAgent):
    """You are helpful."""

    class Config:
        text_format = MyOutput  # Has large fields

    def format_llm_message(self, response) -> str:
        # Only store essential content in memory
        if response.output_parsed:
            return response.output_parsed.summary  # Not the large data!
        return response.output_text or ""
```

### Dependency Injection

```python
class MyAgent(OpenAIAgent):
    """You help users with data."""

    def __init__(self, db_client):
        super().__init__()  # Always call super!
        self.db = db_client

    async def get_user(self, user_id: str) -> dict:
        return await self.db.fetch(user_id)

# Usage
agent = MyAgent(db_client=my_database)
```

### Custom OpenAI Client

```python
from openai import AsyncOpenAI
from pyaiagent import set_default_openai_client

# Configure before using any agent
client = AsyncOpenAI(
    api_key="sk-...",
    base_url="https://your-proxy.com/v1",  # Custom endpoint
    timeout=60.0,
    max_retries=3,
)
set_default_openai_client(client)

# Now agents will use this client
agent = MyAgent()
```

### Local LLMs (Ollama)

```python
from openai import AsyncOpenAI
from pyaiagent import set_default_openai_client

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
set_default_openai_client(client)

class MyAgent(OpenAIAgent):
    """You are helpful."""
    
    class Config:
        model = "llama3.2"  # Your local model
```

---

## Learning Path

**New to pyaiagent?** Follow this order:

1. **Start here:** `00_basic_agent.py` — understand the core concept
2. **Add memory:** `01_conversation_memory.py` — multi-turn chats
3. **Add tools:** `02_tools_basic.py` — make agents do things
4. **Type safety:** `03_structured_output.py` — Pydantic responses
5. **Customize:** `04_config_advanced.py` — tune behavior
6. **Go to production:** `07_fastapi_with_lifespan.py` — real APIs

**Building something complex?**

- `05_dynamic_instructions.py` — personalization
- `08_error_handling.py` — robust error handling
- `09_inheritance_composition.py` — agent hierarchies
- `11_message_formatting.py` — token optimization
- `12_dependency_injection.py` — DB/API client injection
- `13_custom_client.py` — Azure, Ollama, proxies, custom endpoints

---

## Tips

1. **Start simple** — Basic agents work great. Add complexity only when needed.

2. **Reuse agents** — In servers, create once and reuse for all requests.

3. **Write good docstrings** — The AI reads them. Be specific.

4. **Use type hints** — They become the tool parameter schema.

5. **Handle errors** — See `08_error_handling.py` for patterns.

6. **Call `shutdown()`** — In scripts and on server shutdown, clean up properly.

---

## Instance Variables vs Instruction Params

Understanding when to use `__init__` with instance variables versus `instruction_params` is key to building efficient agents.

### When to Use What

| Approach | Use Case | Example |
|----------|----------|---------|
| **Instance variables (`__init__`)** | Static, per-instance dependencies that don't change between requests | DB connections, API clients, config objects |
| **`instruction_params`** | Dynamic, per-request data that changes with each call | User name, current date, user preferences |
| **Single instance + `instruction_params`** | Same agent behavior, different context per request | Multi-tenant apps, personalized responses |

### Pattern 1: Dependency Injection via `__init__`

Use `__init__` when your agent needs external services or static configuration:

```python
class DatabaseAgent(OpenAIAgent):
    """You are a data assistant. Query the database to help users."""

    def __init__(self, db_connection, cache_client=None):
        super().__init__()  # Always call super().__init__()
        self.db = db_connection
        self.cache = cache_client

    async def query_users(self, user_id: str) -> dict:
        """Look up a user by ID."""
        # Use the injected dependency
        return await self.db.fetch_user(user_id)

# Usage
db = DatabaseConnection("postgresql://...")
agent = DatabaseAgent(db_connection=db)
result = await agent.process(input="Find user #123")
```

### Pattern 2: Per-Request Context via `instruction_params`

Use `instruction_params` for data that changes with each request:

```python
class PersonalizedAgent(OpenAIAgent):
    """
    You are helping {user_name}.
    Their preferences: {preferences}
    Today is {date}.
    """

# Same agent instance, different context per request
agent = PersonalizedAgent()
```

**Placeholder behavior:** By default, unmatched `{placeholders}` are left as-is. This is safe for instructions containing example formats. To require all placeholders:

```python
class StrictAgent(OpenAIAgent):
    """You are helping {user_name}."""

    class Config:
        strict_instruction_params = True  # Raises InstructionKeyError if missing
```

```python
# Request from User A
result = await agent.process(
    input="What should I do today?",
    instruction_params={
        "user_name": "Alice",
        "preferences": "loves hiking",
        "date": "Monday, Jan 13"
    }
)

# Request from User B (same agent instance!)
result = await agent.process(
    input="Suggest a movie",
    instruction_params={
        "user_name": "Bob",
        "preferences": "sci-fi fan",
        "date": "Monday, Jan 13"
    }
)
```

### Pattern 3: Single Instance for Production

For web servers, create **one agent instance** and reuse it for all requests:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

class AssistantAgent(OpenAIAgent):
    """You help users with {task_type}."""

    def __init__(self, api_client):
        super().__init__()
        self.api = api_client  # Shared across all requests

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create ONCE at startup
    app.state.agent = AssistantAgent(api_client=MyAPIClient())
    yield
    await shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    agent = req.app.state.agent  # Reuse the same instance
    
    # Customize per-request with instruction_params
    return await agent.process(
        input=request.message,
        instruction_params={"task_type": request.task_type}
    )
```

### Decision Guide

**Use `__init__` + instance variables when:**
- You need database connections, API clients, or external services
- The dependency is expensive to create (connection pools, clients)
- The data is the same for all requests from this agent instance

**Use `instruction_params` when:**
- Data varies per request (user info, current time, context)
- You want one agent to serve multiple users/tenants
- The customization is about "what the agent knows" not "what it can do"

**Use a single instance when:**
- Running in a web server (FastAPI, etc.)
- All requests share the same tools and dependencies
- You want efficient resource usage

**Use multiple instances when:**
- Different instances need different injected dependencies
- You're testing with mock dependencies
- Instances need isolated state (rare)

---

## Need Help?

- 📖 [Full documentation](../README.md)
- 🐛 [Report issues](https://github.com/troymjose/pyaiagent/issues)
- 💬 Questions? Open a discussion!

Happy building! 🚀

