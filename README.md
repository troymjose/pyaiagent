# PyAiAgent

[![PyPI version](https://img.shields.io/pypi/v/pyaiagent.svg)](https://pypi.org/project/pyaiagent/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyaiagent.svg)](https://pypi.org/project/pyaiagent/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/troymjose/pyaiagent/blob/master/LICENSE)
<!-- [![GitHub stars](https://img.shields.io/github/stars/troymjose/pyaiagent.svg?style=social)](https://github.com/troymjose/pyaiagent) -->

PyAiAgent is a modern, fast (high-performance), async framework for building AI agents with pythonic code.

```python
from pyaiagent import OpenAIAgent


class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""


agent = MyAgent()

result = await agent.process(input="Did I just build an AI agent in 2 lines?")
```

---

## Contents

- [Why pyaiagent?](#why-pyaiagent)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Adding Tools](#adding-tools)
- [Configuration](#configuration)
- [Structured Output](#structured-output)
- [Sessions and Conversation Memory](#sessions-and-conversation-memory)
- [Dynamic Instructions](#dynamic-instructions)
- [Dependency Injection](#dependency-injection)
- [Inheritance and Composition](#inheritance-and-composition)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

---

## Why pyaiagent?

- Minimal API – subclass OpenAIAgent, write a docstring, add async methods as tools.
- No magic – no decorators, no YAML, no custom DSL.
- Async‑native – designed for asyncio, FastAPI, and modern Python apps.

### See the Difference

Here's a weather agent with one tool. First, without pyaiagent:

<details>
<summary><b>Without pyaiagent — Raw OpenAI API (~50 lines)</b></summary>

```python
import asyncio
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Manual tool schema — you write this for every tool
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"]
        }
    }
}]


def get_weather(city: str) -> dict:
    return {"city": city, "temperature": "22°C", "condition": "Sunny"}


async def run_agent(user_input: str) -> str:
    messages = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": user_input}
    ]

    while True:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = get_weather(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            return message.content


asyncio.run(run_agent("What's the weather in Paris?"))
```

</details>

**With pyaiagent — 8 lines:**

```python
import asyncio
from pyaiagent import OpenAIAgent


class WeatherAgent(OpenAIAgent):
    """You are a weather assistant."""

    async def get_weather(self, city: str) -> dict:
        """Get the current weather for a city."""
        return {"city": city, "temperature": "22°C", "condition": "Sunny"}


asyncio.run(WeatherAgent().process(input="What's the weather in Paris?"))
```

### How 45 Lines Became 8

Here's exactly what pyaiagent handles for you:

| What You Write | What pyaiagent Does For You |
|----------------|---------------------------|
| `class WeatherAgent(OpenAIAgent):` | Creates the agent with all OpenAI wiring |
| `"""You are a weather assistant."""` | Becomes the system prompt — no `{"role": "system", ...}` dict |
| `async def get_weather(self, city: str)` | Auto-generates the full JSON Schema from type hints |
| `"""Get the current weather..."""` | Becomes the tool description — no manual schema writing |
| `await agent.process(input=...)` | Runs the entire agentic loop — tool detection, execution, response |

**The agentic loop alone saves ~20 lines.** pyaiagent handles:
- Detecting when the AI wants to call tools
- Parsing tool call arguments from JSON
- Executing your tool methods (async or sync)
- Running multiple tools in parallel when the AI requests them
- Formatting results back to the AI
- Looping until the AI produces a final response
- Token counting, error handling, and timeouts

### Simply Pythonic, Fully Flexible

pyaiagent removes boilerplate, **not capabilities**. You still have full access to everything:

```python
class MyAgent(OpenAIAgent):
    """You are a helpful assistant for {user_name}."""  # Dynamic instructions

    class Config:
        model = "gpt-4o"              # Any OpenAI model
        temperature = 0.7             # All generation parameters
        max_output_tokens = 4096      # Response length control
        tool_timeout = 60.0           # Per-tool timeout
        parallel_tool_calls = True    # Parallel execution

    def __init__(self, db):           # Dependency injection
        super().__init__()
        self.db = db

    async def query(self, sql: str) -> dict:  # Tools are just methods
        """Run a database query."""
        return await self.db.execute(sql)
```

**What makes it Pythonic:**
- **Classes** — Agents are classes, not decorated functions or YAML configs
- **Docstrings** — Instructions and tool descriptions are docstrings, not string constants
- **Type hints** — Parameter types are Python types, not JSON Schema
- **Inheritance** — Build specialized agents from base agents using normal inheritance
- **`async`/`await`** — Native async, not callbacks or bolted-on wrappers

**What you're NOT giving up:**
- Custom OpenAI clients (Azure, proxies, local LLMs)
- Structured outputs with Pydantic models
- Multi-turn conversation memory
- Dependency injection for databases, APIs, etc.
- Full control over message formatting
- Access to token usage, step counts, and metadata

The raw OpenAI API is powerful. pyaiagent just removes the parts you rewrite for every agent.

| Feature                  | pyaiagent                             | Other Frameworks                 |
|--------------------------|---------------------------------------|----------------------------------|
| Lines to define an agent | ~8                                    | ~45+                             |
| Learning curve           | Minutes                               | Hours/Days                       |
| Pythonic                 | Yes — classes, docstrings, type hints | Custom DSLs, decorators, configs |
| Decorators needed        | None                                  | Many                             |
| Async support            | Native                                | Often bolted-on                  |
| Dependencies             | 2 packages                            | 50+ packages                     |

**pyaiagent** is for developers who want to build AI agents without wrestling with complex abstractions, heavy
dependencies, or verbose boilerplate.

---

## Installation

```bash
pip install pyaiagent
```

### Requirements

- Python 3.10+
- OpenAI API key

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Quick Start

### Step 1: Create an Agent

Create a file called `my_agent.py`:

```python
from pyaiagent import OpenAIAgent


class MyAgent(OpenAIAgent):
    """
    You are a friendly assistant who helps users with their questions.
    Always be polite and helpful.
    """
```

That's it! The docstring becomes your agent's instructions.

### Step 2: Run the Agent

```python
import asyncio
from my_agent import MyAgent


async def main():
    agent = MyAgent()
    result = await agent.process(input="Is creating an AI agent really this simple?")
    print(result["output"])


asyncio.run(main())
```

---

## Adding Tools

Tools are methods on your agent class. The method name becomes the tool name, and the docstring becomes the tool
description. You can use **async** or **sync** methods depending on your use case.

```python
from pyaiagent import OpenAIAgent


class WeatherAgent(OpenAIAgent):
    """
    You are a weather assistant. Use the get_weather tool 
    to fetch current weather for any city.
    """

    async def get_weather(self, city: str) -> dict:
        """Get the current weather for a city."""
        # In real code, you'd call a weather API here
        return {
            "city": city,
            "temperature": "22°C",
            "condition": "Sunny"
        }
```

### Async vs Sync Tools

| Tool Type | Syntax | Best For | Execution |
|-----------|--------|----------|-----------|
| **Async** | `async def` | I/O-bound (API calls, DB queries) | Direct await |
| **Sync** | `def` | CPU-bound (computation, parsing) | Thread pool (non-blocking) |

```python
class MyAgent(OpenAIAgent):
    """Agent with both async and sync tools."""

    # Async tool: for I/O-bound work (API calls, database)
    async def fetch_data(self, url: str) -> dict:
        """Fetch data from an API."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

    # Sync tool: for CPU-bound work (runs in thread pool automatically)
    def calculate_stats(self, numbers: list[float]) -> dict:
        """Calculate statistics on a list of numbers."""
        import statistics
        return {
            "mean": statistics.mean(numbers),
            "median": statistics.median(numbers),
            "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0
        }
```

Sync tools are automatically run in a thread pool via `asyncio.to_thread()`, so they don't block the event loop.

### How It Works

1. You define a method with a docstring (async or sync)
2. pyaiagent automatically creates a tool schema for OpenAI
3. When the AI decides to use the tool, pyaiagent calls your method
4. The return value is sent back to the AI

### Tool Parameters

Python type hints are automatically converted to JSON Schema:

```python
async def search_products(
    self,
    query: str,                    # Required string
    category: str = None,          # Optional string
    max_price: float = 100.0,      # Optional with default
    in_stock: bool = True          # Optional boolean
) -> dict:
    """Search for products in the catalog."""
    ...
```

### Supported Types

| Python Type         | JSON Schema                                       |
|---------------------|---------------------------------------------------|
| `str`               | `"type": "string"`                                |
| `int`               | `"type": "integer"`                               |
| `float`             | `"type": "number"`                                |
| `bool`              | `"type": "boolean"`                               |
| `list[str]`         | `"type": "array", "items": {"type": "string"}`    |
| `dict[str, int]`    | `"type": "object", "additionalProperties": {...}` |
| `datetime`          | `"type": "string", "format": "date-time"`         |
| `Literal["a", "b"]` | `"enum": ["a", "b"]`                              |
| `Optional[str]`     | `"anyOf": [{"type": "string"}, {"type": "null"}]` |
| `TypedDict`         | Full object schema with properties                |
| `dataclass`         | Full object schema with properties                |
| `Enum`              | Enum values                                       |

---

## Configuration

Customize your agent with a nested `Config` class:

```python
class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""

    class Config:
        model = "gpt-4o"              # OpenAI model to use
        temperature = 0.7             # Creativity (0.0 - 2.0)
        max_output_tokens = 4096      # Max response length
```

### All Configuration Options

| Option                      | Type        | Default         | Description                                        |
|-----------------------------|-------------|-----------------|---------------------------------------------------|
| `model`                     | `str`       | `"gpt-4o-mini"` | OpenAI model ID                                   |
| `temperature`               | `float`     | `0.2`           | Response randomness (0.0-2.0)                     |
| `top_p`                     | `float`     | `None`          | Nucleus sampling (alternative to temperature)     |
| `max_output_tokens`         | `int`       | `4096`          | Maximum tokens in response                        |
| `seed`                      | `int`       | `None`          | For reproducible outputs                          |
| `tool_choice`               | `str`       | `"auto"`        | `"auto"`, `"none"`, or `"required"`               |
| `parallel_tool_calls`       | `bool`      | `True`          | Allow multiple tools at once                      |
| `max_steps`                 | `int`       | `10`            | Max tool-call rounds per request                  |
| `max_parallel_tools`        | `int`       | `10`            | Concurrency limit for tool execution              |
| `tool_timeout`              | `float`     | `30.0`          | Timeout per tool call (seconds)                   |
| `llm_timeout`               | `float`     | `120.0`         | Timeout for LLM response (seconds)                |
| `text_format`               | `BaseModel` | `None`          | Pydantic model for structured output              |
| `strict_instruction_params` | `bool`      | `False`         | Raise error on missing `{placeholder}` params     |
| `validation_retries`        | `int`       | `0`             | Retry count for structured output validation failures (0 = disabled) |

### OpenAI Client Configuration

#### Using Environment Variables

The agent uses the standard OpenAI environment variables:

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export OPENAI_ORG_ID="org-..."                      # Organization ID
export OPENAI_PROJECT_ID="proj-..."                 # Project ID
export OPENAI_BASE_URL="https://your-proxy.com/v1"  # Custom endpoint / proxy
export OPENAI_TIMEOUT="60"                          # Request timeout (seconds)
export OPENAI_MAX_RETRIES="3"                       # Max retry attempts
```

#### Programmatic Configuration

For full control over the OpenAI client, use `set_default_openai_client()` to pass a pre-configured `AsyncOpenAI` client:

```python
from openai import AsyncOpenAI
from pyaiagent import set_default_openai_client, OpenAIAgent

# Create a custom client with full control over all parameters
custom_client = AsyncOpenAI(
    api_key="sk-...",
    base_url="https://your-proxy.com/v1",
    timeout=60.0,
    max_retries=3,
)

# Set it as the default for all agents
set_default_openai_client(custom_client)


# Now create and use agents - they'll use this client
class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""


agent = MyAgent()
result = await agent.process(input="Hello!")
```

**Important:** Call `set_default_openai_client()` **before** using any agent. Once an agent makes its first request, the client is locked in for that event loop.

This approach gives you access to **all** `AsyncOpenAI` parameters, including:

| Parameter | Description |
|-----------|-------------|
| `api_key` | API key (string or async callable) |
| `organization` | Organization ID |
| `project` | Project ID |
| `base_url` | Custom API endpoint (proxies, Azure, local LLMs) |
| `timeout` | Request timeout (seconds or `Timeout` object) |
| `max_retries` | Max retry attempts |
| `default_headers` | Custom headers for all requests |
| `http_client` | Custom `httpx.AsyncClient` for advanced networking |

#### Using Azure OpenAI

**Option 1: Environment Variables**

```bash
export OPENAI_API_KEY="your-azure-key"
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
export OPENAI_API_VERSION="2024-02-01"
```

**Option 2: Programmatic**

```python
from openai import AsyncAzureOpenAI
from pyaiagent import set_default_openai_client

azure_client = AsyncAzureOpenAI(
    api_key="your-azure-key",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com",
)

set_default_openai_client(azure_client)
```

#### Using Local LLMs (Ollama, LM Studio, etc.)

```python
from openai import AsyncOpenAI
from pyaiagent import set_default_openai_client

# Ollama
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Required but not used
)
set_default_openai_client(client)


class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""

    class Config:
        model = "llama3.2"  # Use your local model name
```

---

## Structured Output

Get responses as Pydantic models instead of plain text:

```python
from pydantic import BaseModel
from pyaiagent import OpenAIAgent


class MovieReview(BaseModel):
    title: str
    rating: int  # 1-10
    summary: str
    recommended: bool


class ReviewAgent(OpenAIAgent):
    """
    You are a movie critic. Analyze movies and provide structured reviews.
    """

    class Config:
        model = "gpt-4o"
        text_format = MovieReview


agent = ReviewAgent()
result = await agent.process(input="Review the movie Inception")

# Parsed Pydantic model
review = result["output_parsed"]
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Recommended: {review.recommended}")
```

### Validation Retries

OpenAI's Structured Outputs guarantees JSON schema conformance, but **custom Pydantic validators** (business rules, cross-field checks, semantic constraints) can still fail. When they do, pyaiagent can automatically retry by sending the validation errors back to the LLM, giving it a chance to self-correct.

```python
from pydantic import BaseModel, field_validator
from pyaiagent import OpenAIAgent


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

    @field_validator("rating")
    @classmethod
    def rating_must_be_valid(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Rating must be between 1 and 10")
        return v

    @field_validator("summary")
    @classmethod
    def summary_must_be_detailed(cls, v):
        if len(v.split()) < 10:
            raise ValueError("Summary must be at least 10 words")
        return v


class ReviewAgent(OpenAIAgent):
    """You are a movie critic. Provide detailed, structured reviews."""

    class Config:
        text_format = MovieReview
        validation_retries = 3  # Retry up to 3 times on validation failure
```

**How it works:**

1. The LLM produces a structured response (JSON schema enforced with `strict: True`)
2. Pydantic validates it (including your custom validators)
3. If validation fails, the specific errors are sent back to the LLM for self-correction
4. The LLM corrects its output and tries again
5. This repeats up to `validation_retries` times
6. If all retries are exhausted, `ValidationRetriesExhaustedError` is raised

**Token tracking** is fully accurate during validation retries. Every retry attempt — including failed ones — contributes to the `input_tokens`, `output_tokens`, and `total_tokens` reported in the result (or on exceptions via `e.tokens`). This makes per-customer cost attribution reliable even in multi-tenant systems.

**Configuration:**

| Value | Behavior |
|---|---|
| `validation_retries = 0` | **Default.** No retries. Current behavior preserved. Use this if you handle retries yourself. |
| `validation_retries = 3` | Retry up to 3 times. Recommended for most use cases. |

**What happens to messages during retries:**

Retry artifacts (failed responses and error feedback) are handled differently in each message list:

- **`history`** — **Clean.** All retry artifacts are automatically removed before returning. The returned list looks as if the retry never happened — only the final valid response is included. This means no wasted tokens when you pass `history` to the next `process()` call.

- **`messages`** — **Full history preserved.** Every step is visible, including failed attempts. Each message has a `step` number, so you can see exactly which step was a retry. This is useful for debugging, analytics, and audit logs.

```
Example: validation_retries=3, fails once then succeeds

history (returned to caller):
  [...history, user_input, corrected_assistant]     ← clean, retry invisible

messages (returned to caller):
  [{role: "user",      content: "Review Inception",    step: 1, tokens: {...}},
   {role: "assistant", content: '{"rating": 0, ...}',  step: 1, tokens: {...}},  ← failed attempt
   {role: "assistant", content: '{"rating": 8, ...}',  step: 2, tokens: {...}}]  ← corrected response
```

This separation ensures that the LLM never sees old validation noise in subsequent turns, while your application retains full visibility into agent behavior.

**Error handling:**

```python
from pyaiagent import ValidationRetriesExhaustedError

try:
    result = await agent.process(input="Review Inception")
except ValidationRetriesExhaustedError as e:
    print(f"Validation failed after retries: {e}")
    print(f"Last errors: {e.validation_errors}")
```

---

## Sessions and Conversation Memory

By default, each `process()` call is independent — the agent doesn't remember previous messages.

To create a multi-turn conversation, pass the previous messages back:

```python
agent = MyAgent()

# Turn 1: User introduces themselves
result1 = await agent.process(input="My name is Alice")

# Turn 2: Pass previous messages so the agent remembers
result2 = await agent.process(
    input="What's my name?",
    history=result1["history"]  # ← This enables memory
)

print(result2["output"])  # "Your name is Alice"
```

**How it works:**

1. `result1["history"]` contains the conversation history
2. Pass it to the next `process()` call via `history`
3. The agent now "remembers" the previous conversation

**Tip:** For longer conversations, keep updating the messages:

```python
history = []

for user_input in ["Hi, I'm Alice", "What's my name?", "Thanks!"]:
    result = await agent.process(input=user_input, history=history)
    history = result["history"]
    print(result["output"])
```

**Token optimization:** When using structured outputs with large fields, conversation memory can grow quickly. Override `format_history()` to control what gets stored. See [Best Practices #6](#6-customize-message-storage-token-optimization) for details.

### Understanding `history` vs `messages`

Every `process()` call returns two message lists, each designed for a different purpose:

| | `history` | `messages` |
|---|---|---|
| **Contains** | Full accumulated conversation history | Only the current turn's messages |
| **Purpose** | Feed back into the next `process()` call so the LLM has context | Display in chat UI, store as audit log |
| **Grows across turns?** | Yes — includes all previous messages + current turn | No — always scoped to this single `process()` call |
| **Validation retries** | Clean — retry artifacts are removed, only the final valid response is kept | Full history — shows all attempts including failures (with `step` numbers) |
| **DB pattern** | **Overwrite** the session record each request | **Insert/append** to a messages collection each request |

**`history`** is the LLM's working memory. It accumulates the full conversation (user messages, assistant responses, tool calls, tool results) because the OpenAI API needs the complete history to maintain context. When you pass it back via `history`, the agent picks up right where it left off. If validation retries occurred, the failed attempts and error feedback are automatically removed — only the final valid response remains, so no tokens are wasted in subsequent turns.

**`messages`** is your application's structured record of what happened in *this turn only*. Each message is enriched with metadata (`agent`, `session`, `turn`, `step`, `tokens`) making it ready for direct insertion into a database or display in a chat interface. If validation retries occurred, all attempts (including failures) are preserved for full debugging and analytics visibility.

```
Turn 1:  process(input="Hi, I'm Alice")
         history = [user_msg_1, assistant_msg_1]          ← 2 messages
         messages  = [user_msg_1, assistant_msg_1]          ← 2 messages

Turn 2:  process(input="What's my name?", history=...)
         history = [user_1, asst_1, user_2, asst_2]      ← 4 messages (accumulated)
         messages  = [user_msg_2, assistant_msg_2]          ← 2 messages (current turn only)
```

### Production Session Management

The agent is **stateless by design** — it never stores conversation history internally. This means session management in production is straightforward: just load and save a single JSON list per session.

#### The Pattern (3 Steps)

```python
@app.post("/chat")
async def chat(session_id: str, message: str):
    # 1. LOAD — Get the conversation history for this session
    history = await db.load_session(session_id)   # [] if new session

    # 2. PROCESS — The agent handles everything
    result = await agent.process(
        input=message,
        session=session_id,
        history=history
    )

    # 3. SAVE — Overwrite history, insert messages
    await db.save_session(session_id, result["history"])    # Overwrite
    await db.insert_events(session_id, result["events"])      # Append

    return {"response": result["output"]}
```

That's it. Three steps: **load, process, save**.

#### Database Schema

```
sessions table (overwritten each request)
┌──────────────┬──────────────────────────────┬────────────┐
│ session_id   │ history (JSONB)         │ updated_at │
│ "user-123"   │ [full conversation history]  │ 2025-01-15 │
└──────────────┴──────────────────────────────┴────────────┘

messages table (append-only, grows over time)
┌────┬────────────┬──────┬──────┬──────┬──────────┬────────┐
│ id │ session_id │ turn │ step │ role │ content  │ tokens │
│ 1  │ user-123   │ t1   │ 1    │ user │ "Hi..."  │ {...}  │
│ 2  │ user-123   │ t1   │ 1    │ asst │ "Hello!" │ {...}  │
│ 3  │ user-123   │ t2   │ 1    │ user │ "Name?"  │ {...}  │  ← new turn appended
│ 4  │ user-123   │ t2   │ 1    │ asst │ "Alice!" │ {...}  │
└────┴────────────┴──────┴──────┴──────┴──────────┴────────┘
```

#### With Redis (Fast Sessions + TTL)

```python
import json
import redis.asyncio as redis

@app.post("/chat")
async def chat(session_id: str, message: str):
    # 1. LOAD
    raw = await redis_client.get(f"session:{session_id}")
    history = json.loads(raw) if raw else []

    # 2. PROCESS
    result = await agent.process(input=message, session=session_id, history=history)

    # 3. SAVE — overwrite with TTL (auto-expires abandoned sessions)
    await redis_client.setex(f"session:{session_id}", 3600, json.dumps(result["history"]))

    return {"response": result["output"]}
```

#### With PostgreSQL (Persistent History)

```python
@app.post("/chat")
async def chat(session_id: str, message: str):
    # 1. LOAD
    row = await db.fetchrow("SELECT history FROM sessions WHERE session_id = $1", session_id)
    history = row["history"] if row else []

    # 2. PROCESS
    result = await agent.process(input=message, session=session_id, history=history)

    # 3. SAVE — upsert session, insert UI messages
    await db.execute("""
        INSERT INTO sessions (session_id, history, updated_at) VALUES ($1, $2, NOW())
        ON CONFLICT (session_id) DO UPDATE SET history = $2, updated_at = NOW()
    """, session_id, json.dumps(result["history"]))

    await db.executemany(
        "INSERT INTO messages (session_id, data) VALUES ($1, $2)",
        [(session_id, json.dumps(evt)) for evt in result["events"]]
    )

    return {"response": result["output"]}
```

#### Production Tips

- **Limit conversation length** — The `history` list grows with every turn. Truncate old messages to control token usage:
  ```python
  MAX_MESSAGES = 40
  if len(history) > MAX_MESSAGES:
      history = history[-MAX_MESSAGES:]
  ```
- **Session expiry** — Use Redis TTL or a scheduled cleanup job to remove abandoned sessions.
- **Concurrency** — If a session can receive concurrent requests, use a distributed lock to prevent race conditions.

### Response Structure

```python
result = {
    "input": "What's my name?",
    "output": "Your name is Alice",
    "output_parsed": None,  # Pydantic model if text_format is set
    "session": "user-123",
    "turn": "uuid-of-this-turn",
    "steps": 1,
    "tokens": {
        "input_tokens": 25,
        "output_tokens": 8,
        "total_tokens": 33
    },
    "history": [...],   # Full conversation history — overwrite in DB, pass to next process()
    "events": [...],    # Current turn only — append to DB, display in chat UI
    "metadata": {}
}
```

---

## Dynamic Instructions

Use placeholders in your instructions:

```python
class PersonalizedAgent(OpenAIAgent):
    """
    You are a personal assistant for {user_name}.
    Their preferences: {preferences}
    Today's date is {date}.
    """


agent = PersonalizedAgent()
result = await agent.process(
    input="What should I do today?",
    instruction_params={
        "user_name": "Alice",
        "preferences": "loves hiking, vegetarian",
        "date": "2025-01-15"
    }
)
```

### Placeholder Behavior

By default, unmatched `{placeholders}` are left as-is. This is useful when your instructions contain example formats or code snippets:

```python
class MyAgent(OpenAIAgent):
    """
    You are an assistant for {user_name}.
    Return responses in this format: {field}: value
    """

# Only {user_name} is replaced; {field} stays as literal text
result = await agent.process(
    input="Hello",
    instruction_params={"user_name": "Alice"}
)
```

To enforce that all placeholders must be provided, enable strict mode:

```python
class StrictAgent(OpenAIAgent):
    """You are an assistant for {user_name}."""

    class Config:
        strict_instruction_params = True  # Raises InstructionKeyError if {user_name} is missing
```

---

## Dependency Injection

Agents can accept dependencies via `__init__` for static, per-instance configuration:

```python
class DatabaseAgent(OpenAIAgent):
    """You are a data assistant."""

    def __init__(self, db_connection):
        super().__init__()  # Always call super().__init__()
        self.db = db_connection

    async def query_users(self, user_id: str) -> dict:
        """Look up a user by ID."""
        return await self.db.fetch_user(user_id)


# Usage
db = DatabaseConnection("postgresql://...")
agent = DatabaseAgent(db_connection=db)
```

### When to Use What

| Approach | Use Case | Lifecycle |
|----------|----------|-----------|
| `__init__` + instance variables | DB connections, API clients, static config | Set once at instantiation |
| `instruction_params` | User name, date, preferences, context | Changes per `process()` call |

**Rule of thumb:**
- `__init__` is for "what the agent **has**" (dependencies, clients)
- `instruction_params` is for "what the agent **knows**" (context, user info)

For production servers, combine both patterns — create one agent with injected dependencies at startup, and customize per-request with `instruction_params`:

```python
# Create once at startup with dependencies
agent = MyAgent(db_connection=db, api_client=client)

# Customize per-request with instruction_params
result = await agent.process(
    input=user_message,
    instruction_params={"user_name": current_user.name}
)
```

---

## Inheritance and Composition

Build specialized agents from base agents:

```python
class BaseAssistant(OpenAIAgent):
    """You are a helpful assistant."""

    async def get_time(self) -> dict:
        """Get the current time."""
        from datetime import datetime
        return {"time": datetime.now().isoformat()}


class CustomerSupportAgent(BaseAssistant):
    """
    You are a customer support agent for Acme Inc.
    Be professional and helpful. You can check the time if needed.
    """

    class Config:
        model = "gpt-4o"
        temperature = 0.3

    async def lookup_order(self, order_id: str) -> dict:
        """Look up an order by ID."""
        return {"order_id": order_id, "status": "shipped"}
```

`CustomerSupportAgent` inherits:

- The `get_time` tool from `BaseAssistant`
- Can override config and add new tools

---

## Error Handling

All agent process exceptions inherit from `OpenAIAgentProcessError`. You can catch specific errors or use the base class
to catch all:

```python
from pyaiagent import (
    OpenAIAgentProcessError,  # Base class - catches all agent process errors
    MaxStepsExceededError,
    ClientError,
)

agent = MyAgent()

try:
    result = await agent.process(input="Hello")
except MaxStepsExceededError as e:
    print(f"Agent took too many steps | tokens consumed: {e.tokens}")
except ClientError as e:
    print(f"OpenAI API error: {e} | tokens consumed: {e.tokens}")
except OpenAIAgentProcessError as e:
    # Catches any other agent process error
    print(f"Agent error: {e}")
```

### Token Tracking on Errors

Every exception raised during processing carries a `tokens` attribute — a dict with `input_tokens`, `output_tokens`,
and `total_tokens` consumed before the error occurred. This ensures you always have cost visibility, even on failure:

```python
try:
    result = await agent.process(input="Complex question...")
except OpenAIAgentProcessError as e:
    print(e.tokens)
    # {"input_tokens": 1250, "output_tokens": 340, "total_tokens": 1590}
```

For input validation errors raised before any API call (e.g., `InvalidInputError`), `tokens` is `None`.

### Exception Types

| Exception                       | When                                                | `tokens` |
|---------------------------------|-----------------------------------------------------|----------|
| `InvalidInputError`             | `input` is not a string                             | `None`   |
| `InvalidSessionError`           | `session` is empty or not a string                  | `None`   |
| `InvalidMetadataError`          | `metadata` is not a dict                            | `None`   |
| `InvalidHistoryError`       | `history` is not a list                        | `None`   |
| `InvalidInstructionParamsError` | `instruction_params` is not a dict                  | `None`   |
| `InstructionKeyError`           | Missing placeholder key (only if `strict_instruction_params`) | `None` |
| `ClientError`                   | OpenAI API returned an error                        | `dict`   |
| `MaxStepsExceededError`         | Agent exceeded `max_steps` without completing       | `dict`   |
| `ValidationRetriesExhaustedError` | Structured output validation failed after all retry attempts | `dict` |
| `OpenAIAgentClosedError`        | Agent used after `aclose()` called                  | `None`   |

---

## Best Practices

### 1. Reuse Agents in Servers

For FastAPI or other servers, create the agent **once** and reuse it for all requests:

```python
from fastapi import FastAPI

agent = MyAgent()  # Create once at module level
app = FastAPI()


@app.post("/chat")
async def chat(message: str):
    # Reuse the same agent for every request
    result = await agent.process(input=message)
    return {"response": result["output"]}
```

**For proper cleanup on shutdown**, use the lifespan pattern:

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pyaiagent import shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = MyAgent()
    yield
    await shutdown()  # Cleanup shared OpenAI client on shutdown


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(message: str):
    result = await app.state.agent.process(input=message)
    return {"response": result["output"]}
```

### 2. Write Clear Docstrings

```python
# ✅ Good - clear instruction
class MyAgent(OpenAIAgent):
    """
    You are a travel booking assistant for SkyHigh Airlines.
    Help users find and book flights. Be friendly and professional.
    Always confirm details before booking.
    """


# ❌ Bad - vague instruction
class MyAgent(OpenAIAgent):
    """Assistant."""
```

### 3. Use Type Hints for Tools

```python
# ✅ Good - AI knows parameter types
async def search(self, query: str, limit: int = 10) -> dict:
    """Search for items."""
    ...


# ❌ Bad - AI doesn't know types
async def search(self, query, limit):
    """Search for items."""
    ...
```

### 4. Return Dicts from Tools

```python
# ✅ Good - structured response
async def get_user(self, user_id: str) -> dict:
    return {"name": "Alice", "email": "alice@example.com"}


# ⚠️ Works but less informative
async def get_user(self, user_id: str) -> dict:
    return {"result": "Alice"}
```

### 5. Set Appropriate Timeouts

```python
class Config:
    tool_timeout = 60.0   # For slow external APIs
    llm_timeout = 180.0   # For complex reasoning
    max_steps = 5         # Limit runaway loops
```

### 6. Inspect Agent Definitions

Use `get_definition()` to see exactly what pyaiagent auto-generated from your class — the instruction, merged config, and tool schemas as they will be sent to OpenAI:

```python
import json

class MyAgent(OpenAIAgent):
    """You are a helpful assistant for {user_name}."""

    class Config:
        model = "gpt-4o"
        temperature = 0.5

    async def search(self, query: str, limit: int = 10) -> dict:
        """Search the web for information."""
        ...

print(json.dumps(MyAgent.get_definition(), indent=2))
```

This returns:

```json
{
  "agent_name": "MyAgent",
  "instruction": "You are a helpful assistant for {user_name}.",
  "config": {
    "model": "gpt-4o",
    "temperature": 0.5
  },
  "tools": {
    "search": {
      "type": "function",
      "name": "search",
      "description": "Search the web for information.",
      "parameters": { "..." : "..." },
      "strict": true
    }
  }
}
```

This is a classmethod — call it on the class, not an instance. Useful for:
- **Debugging** — verify the instruction text, config values, and tool schemas are correct
- **CI/CD** — snapshot test your agent definitions to catch unintended changes
- **Documentation** — auto-generate tool docs from the schemas

### 7. Customize Message Storage (Token Optimization)

When using structured outputs with large fields, you can reduce token usage by customizing what gets stored in conversation memory:

```python
from pydantic import BaseModel

class MyOutput(BaseModel):
    agent_response: str   # Small - what user sees
    large_data: str       # Large - don't need in memory


class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""

    class Config:
        text_format = MyOutput

    def format_history(self, response) -> str:
        # Only store agent_response in LLM memory (saves tokens!)
        if response.output_parsed:
            return response.output_parsed.agent_response
        return response.output_text or ""

    def format_event(self, response) -> str:
        # Clean, user-friendly view for UI (not raw JSON!)
        if response.output_parsed:
            return response.output_parsed.agent_response
        return response.output_text or ""
```

Both hooks can return the same clean content, or `format_event` can include additional context for display (like timestamps, metadata summaries, etc.) while keeping `format_history` minimal for token efficiency.

**Token savings example:**

| Turns | Without optimization | With optimization |
|-------|---------------------|-------------------|
| 10    | ~50,000 tokens      | ~5,000 tokens     |

---

## API Reference

### `OpenAIAgent`

Base class for all agents.

#### Class Attributes (set automatically)

| Attribute           | Description            |
|---------------------|------------------------|
| `__agent_name__`    | Class name             |
| `__instruction__`   | Processed docstring    |
| `__config_kwargs__` | Merged configuration   |
| `__tool_names__`    | Set of tool names      |
| `__tools_schema__`  | Generated tool schemas |

#### Methods

| Method                 | Description                                    |
|------------------------|------------------------------------------------|
| `async process(...)`   | Process a user input                           |
| `async aclose()`       | Close the agent and release resources          |
| `get_definition()`     | Inspect the auto-generated agent definition (classmethod) |
| `format_history()` | Override to customize LLM message content      |
| `format_event()`       | Override to customize event content             |
| `async __aenter__()`   | Context manager entry                          |
| `async __aexit__(...)`| Context manager exit                           |

### `set_default_openai_client(client)`

Set a custom `AsyncOpenAI` client for all agents to use.

```python
from openai import AsyncOpenAI
from pyaiagent import set_default_openai_client

client = AsyncOpenAI(api_key="sk-...", base_url="https://...")
set_default_openai_client(client)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | `AsyncOpenAI` | A configured OpenAI client instance |

- Must be called **before** any agent is used
- Gives full control over all client parameters
- Works with `AsyncOpenAI`, `AsyncAzureOpenAI`, or compatible clients

### `get_default_openai_client()`

Get the currently configured default client (if any).

```python
from pyaiagent import get_default_openai_client

client = get_default_openai_client()
if client:
    print("Custom client is configured")
else:
    print("Using default client")
```

Returns `AsyncOpenAI | None`.

### `shutdown()`

Gracefully close the shared OpenAI client for the current event loop.

```python
from pyaiagent import shutdown

await shutdown()
```

- **No-op** if no client was ever created on this loop
- **Safe** to call multiple times
- Use in server shutdown handlers (FastAPI lifespan, etc.)

### `get_definition()`

Inspect the auto-generated agent definition. Returns the instruction, config, and tool schemas exactly as they will be sent to OpenAI.

```python
definition = MyAgent.get_definition()
```

#### Return Value

| Key           | Type   | Description                                    |
|---------------|--------|------------------------------------------------|
| `agent_name`  | `str`  | The resolved class name                        |
| `instruction` | `str`  | The cleaned docstring (system prompt)          |
| `config`      | `dict` | Merged config values from parent + child Config |
| `tools`       | `dict` | Tool schemas keyed by tool name                |

This is a **classmethod** — call it on the class (`MyAgent.get_definition()`), not on an instance.

### `process()`

The main method to interact with your agent.

```python
result = await agent.process(
    input="Hello!",
    session="user-123",        # Optional
    history=[...],        # Optional - for conversation memory
    instruction_params={...},  # Optional - for dynamic instructions
    metadata={...}             # Optional - custom data
)
```

#### Parameters

| Parameter            | Type   | Required | Description                                            |
|----------------------|--------|----------|--------------------------------------------------------|
| `input`              | `str`  | Yes      | The user's message to process                          |
| `session`            | `str`  | No       | Session ID for tracking (default: auto-generated UUID) |
| `history`       | `list` | No       | Previous messages for multi-turn conversations         |
| `instruction_params` | `dict` | No       | Values for `{placeholders}` in agent docstring         |
| `metadata`           | `dict` | No       | Custom metadata passed through to response             |

#### Return Value

Returns a dictionary with:

```python
{
    "input": "Hello!",                   # Original input
    "output": "Hi there! How can I...",  # Agent's text response
    "output_parsed": None,               # Pydantic model if text_format is set
    "session": "user-123",               # Session ID
    "turn": "uuid-of-this-turn",         # Unique turn identifier
    "steps": 1,                          # Number of LLM rounds taken
    "tokens": {
        "input_tokens": 25,
        "output_tokens": 42,
        "total_tokens": 67
    },
    "history": [...],                    # Full history — overwrite in DB, pass to next process()
    "events": [...],                     # Current turn only — append to DB, display in chat UI
    "metadata": {}                       # Your custom metadata
}
```

| Key             | Type              | Description                                     |
|-----------------|-------------------|-------------------------------------------------|
| `input`         | `str`             | The original user input                         |
| `output`        | `str`             | The agent's final text response                 |
| `output_parsed` | `BaseModel\|None` | Parsed Pydantic model (if `text_format` is set) |
| `session`       | `str`             | Session identifier                              |
| `turn`          | `str`             | Unique ID for this conversation turn            |
| `steps`         | `int`             | Number of LLM ↔ tool rounds                     |
| `tokens`        | `dict`            | Token usage breakdown                           |
| `history`       | `list`            | Full accumulated conversation history — pass to next `process()` for memory, overwrite in DB per session |
| `messages`      | `list`            | Current turn messages only (enriched with `agent`, `session`, `turn`, `step`, `tokens`) — append to DB, display in UI |
| `metadata`      | `dict`            | Custom metadata passed through                  |

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ for developers who value simplicity.
</p>

