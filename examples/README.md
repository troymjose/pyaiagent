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
| [`02_tools_basic.py`](02_tools_basic.py) | Give agents superpowers | `async` tools, type hints, docstrings |

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

---

## Quick Reference

### Minimal Agent

```python
from pyaiagent import OpenAIAgent

class MyAgent(OpenAIAgent):
    """You are a helpful assistant."""
    pass

agent = MyAgent()
result = await agent.process(input="Hello!")
print(result["output"])
```

### Agent with Tools

```python
class ToolAgent(OpenAIAgent):
    """You can use tools to help users."""

    async def get_weather(self, city: str) -> dict:
        """Get weather for a city."""
        return {"city": city, "temp": "22°C"}
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

---

## Tips

1. **Start simple** — Basic agents work great. Add complexity only when needed.

2. **Reuse agents** — In servers, create once and reuse for all requests.

3. **Write good docstrings** — The AI reads them. Be specific.

4. **Use type hints** — They become the tool parameter schema.

5. **Handle errors** — See `08_error_handling.py` for patterns.

6. **Call `shutdown()`** — In scripts and on server shutdown, clean up properly.

---

## Need Help?

- 📖 [Full documentation](../README.md)
- 🐛 [Report issues](https://github.com/troymjose/pyaiagent/issues)
- 💬 Questions? Open a discussion!

Happy building! 🚀

