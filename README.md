# pyaiagent

A lightweight, high-performance framework for building OpenAI-powered agents in Python.

```python
from pyaiagent import OpenAIAgent

class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""

result = await agent.process(input="Did I just build an AI agent in 2 lines?")
```

---

## Why pyaiagent?

| Feature | pyaiagent | Other Frameworks |
|---------|-----------|------------------|
| Lines to define an agent | ~10 | ~50+ |
| Learning curve | Minutes | Hours/Days |
| Pythonic | Yes — classes, docstrings, type hints | Custom DSLs, decorators, configs |
| Decorators needed | None | Many |
| Async support | Native | Often bolted-on |
| Dependencies | 2 packages | 50+ packages |

**pyaiagent** is for developers who want to build AI agents without wrestling with complex abstractions, heavy dependencies, or verbose boilerplate.

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
    pass
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

Tools are async methods on your agent class. The method name becomes the tool name, and the docstring becomes the tool description.

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

### How It Works

1. You define `async def get_weather(self, city: str)`
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

| Python Type | JSON Schema |
|-------------|-------------|
| `str` | `"type": "string"` |
| `int` | `"type": "integer"` |
| `float` | `"type": "number"` |
| `bool` | `"type": "boolean"` |
| `list[str]` | `"type": "array", "items": {"type": "string"}` |
| `dict[str, int]` | `"type": "object", "additionalProperties": {...}` |
| `datetime` | `"type": "string", "format": "date-time"` |
| `Literal["a", "b"]` | `"enum": ["a", "b"]` |
| `Optional[str]` | `"anyOf": [{"type": "string"}, {"type": "null"}]` |
| `TypedDict` | Full object schema with properties |
| `dataclass` | Full object schema with properties |
| `Enum` | Enum values |

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

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | OpenAI model ID |
| `temperature` | `float` | `0.2` | Response randomness (0.0-2.0) |
| `max_output_tokens` | `int` | `4096` | Maximum tokens in response |
| `tool_choice` | `str` | `"auto"` | `"auto"`, `"none"`, or `"required"` |
| `parallel_tool_calls` | `bool` | `True` | Allow multiple tools at once |
| `max_steps` | `int` | `10` | Max tool-call rounds per request |
| `max_parallel_tools` | `int` | `10` | Concurrency limit for tool execution |
| `tool_timeout` | `float` | `30.0` | Timeout per tool call (seconds) |
| `llm_timeout` | `float` | `120.0` | Timeout for LLM response (seconds) |
| `text_format` | `BaseModel` | `None` | Pydantic model for structured output |

### OpenAI Client Configuration

The agent uses the standard OpenAI environment variables:

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export OPENAI_ORG_ID="org-..."                    # Organization ID
export OPENAI_PROJECT_ID="proj-..."               # Project ID
export OPENAI_BASE_URL="https://your-proxy.com/v1"  # Custom endpoint / proxy
export OPENAI_TIMEOUT="60"                        # Request timeout (seconds)
export OPENAI_MAX_RETRIES="3"                     # Max retry attempts
```

**Using Azure OpenAI:**

```bash
export OPENAI_API_KEY="your-azure-key"
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
export OPENAI_API_VERSION="2024-02-01"            # Azure API version
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
    llm_messages=result1["messages"]["llm"]  # ← This enables memory
)

print(result2["output"])  # "Your name is Alice"
```

**How it works:**
1. `result1["messages"]["llm"]` contains the conversation history
2. Pass it to the next `process()` call via `llm_messages`
3. The agent now "remembers" the previous conversation

**Tip:** For longer conversations, keep updating the messages:

```python
llm_messages = []

for user_input in ["Hi, I'm Alice", "What's my name?", "Thanks!"]:
    result = await agent.process(input=user_input, llm_messages=llm_messages)
    llm_messages = result["messages"]["llm"]
    print(result["output"])
```

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
    "messages": {
        "llm": [...],  # Pass to next turn for memory
        "ui": [...]    # Formatted for display
    },
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

All agent process exceptions inherit from `OpenAIAgentProcessError`. You can catch specific errors or use the base class to catch all:

```python
from pyaiagent import (
    OpenAIAgentProcessError,  # Base class - catches all agent process errors
    MaxStepsExceededError,
    ClientError,
)

agent = MyAgent()

try:
    result = await agent.process(input="Hello")
except MaxStepsExceededError:
    print("Agent took too many steps")
except ClientError as e:
    print(f"OpenAI API error: {e}")
except OpenAIAgentProcessError as e:
    # Catches any other agent process error
    print(f"Agent error: {e}")
```

### Exception Types

| Exception | When |
|-----------|------|
| `InvalidInputError` | `input` is not a string |
| `InvalidSessionError` | `session` is empty or not a string |
| `InvalidMetadataError` | `metadata` is not a dict |
| `InvalidLlmMessagesError` | `llm_messages` is not a list |
| `InvalidInstructionParamsError` | `instruction_params` is not a dict |
| `InstructionKeyError` | Missing key in `instruction_params` for placeholder |
| `ClientError` | OpenAI API returned an error |
| `MaxStepsExceededError` | Agent exceeded `max_steps` without completing |
| `OpenAIAgentClosedError` | Agent used after `aclose()` called |

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = MyAgent()
    yield
    await app.state.agent.aclose()  # Cleanup on shutdown

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

---

## API Reference

### `OpenAIAgent`

Base class for all agents.

#### Class Attributes (set automatically)

| Attribute | Description |
|-----------|-------------|
| `__agent_name__` | Class name |
| `__instruction__` | Processed docstring |
| `__config_kwargs__` | Merged configuration |
| `__tool_names__` | Set of tool names |
| `__tools_schema__` | Generated tool schemas |

#### Methods

| Method | Description |
|--------|-------------|
| `async process(...)` | Process a user input |
| `async aclose()` | Close the agent and release resources |
| `async __aenter__()` | Context manager entry |
| `async __aexit__(...)` | Context manager exit |

#### `process()` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `str` | Yes | User message |
| `session` | `str` | No | Session ID for tracking |
| `llm_messages` | `list` | No | Previous messages for memory |
| `instruction_params` | `dict` | No | Placeholders for instructions |
| `metadata` | `dict` | No | Custom metadata |

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

