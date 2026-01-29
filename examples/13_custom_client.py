"""
Example 13: Custom OpenAI Client Configuration

This example shows how to configure the underlying AsyncOpenAI client
for custom endpoints, proxies, Azure OpenAI, local LLMs, and more.

Key concepts:
- set_default_openai_client() for full client control
- get_default_openai_client() to check current configuration
- Azure OpenAI integration
- Local LLM support (Ollama, LM Studio)
- Custom timeouts and retry settings

Run: python examples/13_custom_client.py
"""

import asyncio
from openai import AsyncOpenAI

from pyaiagent import (
    OpenAIAgent,
    set_default_openai_client,
    get_default_openai_client,
    shutdown,
)


# =============================================================================
# Example 1: Custom Base URL (Proxy or OpenAI-compatible API)
# =============================================================================

def setup_custom_endpoint():
    """
    Use a custom endpoint like a proxy server or OpenAI-compatible API.
    
    This is useful for:
    - Corporate proxies
    - API gateways
    - Self-hosted OpenAI-compatible servers
    """
    client = AsyncOpenAI(
        api_key="your-api-key",
        base_url="https://your-proxy.example.com/v1",
        timeout=60.0,
        max_retries=3,
    )
    set_default_openai_client(client)
    print("✓ Custom endpoint configured")


# =============================================================================
# Example 2: Azure OpenAI
# =============================================================================

def setup_azure_openai():
    """
    Configure for Azure OpenAI Service.
    
    Prerequisites:
    - Azure OpenAI resource created
    - Model deployed in Azure
    """
    # Option 1: Using AsyncAzureOpenAI (recommended)
    from openai import AsyncAzureOpenAI
    
    client = AsyncAzureOpenAI(
        api_key="your-azure-api-key",
        api_version="2024-02-01",
        azure_endpoint="https://your-resource.openai.azure.com",
    )
    set_default_openai_client(client)
    print("✓ Azure OpenAI configured")
    
    # Note: You'll also need to set the model in your agent's Config
    # to match your Azure deployment name


# =============================================================================
# Example 3: Local LLMs (Ollama)
# =============================================================================

def setup_ollama():
    """
    Use Ollama for local LLM inference.
    
    Prerequisites:
    - Ollama installed and running (https://ollama.ai)
    - Model pulled: ollama pull llama3.2
    """
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Required but not validated by Ollama
    )
    set_default_openai_client(client)
    print("✓ Ollama configured")


# =============================================================================
# Example 4: Local LLMs (LM Studio)
# =============================================================================

def setup_lm_studio():
    """
    Use LM Studio for local LLM inference.
    
    Prerequisites:
    - LM Studio installed (https://lmstudio.ai)
    - Model loaded and server started in LM Studio
    """
    client = AsyncOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",  # Required but not validated
    )
    set_default_openai_client(client)
    print("✓ LM Studio configured")


# =============================================================================
# Example 5: Advanced Configuration (Custom Headers, HTTP Client)
# =============================================================================

def setup_advanced():
    """
    Advanced configuration with custom headers and httpx client.
    
    Useful for:
    - Adding authentication headers
    - Custom SSL certificates
    - Proxy configuration at HTTP level
    - Request tracing/logging
    """
    import httpx
    
    # Custom httpx client with specific settings
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        # verify="/path/to/custom/ca-bundle.crt",  # Custom SSL certs
    )
    
    client = AsyncOpenAI(
        api_key="your-api-key",
        default_headers={
            "X-Custom-Header": "my-value",
            "X-Request-ID": "tracking-id",
        },
        http_client=http_client,
    )
    set_default_openai_client(client)
    print("✓ Advanced configuration applied")


# =============================================================================
# Example Agent
# =============================================================================

class AssistantAgent(OpenAIAgent):
    """
    You are a helpful assistant. Be concise and friendly.
    """
    
    class Config:
        model = "gpt-4o-mini"  # Change for Azure/Ollama (e.g., "llama3.2")
        temperature = 0.7


# =============================================================================
# Demo: Check Configuration and Run
# =============================================================================

async def main():
    """
    Demonstrates custom client configuration.
    
    Uncomment ONE of the setup functions below to test different configurations.
    """
    
    # --- Choose your configuration ---
    # setup_custom_endpoint()
    # setup_azure_openai()
    # setup_ollama()
    # setup_lm_studio()
    # setup_advanced()
    
    # --- Check current configuration ---
    current_client = get_default_openai_client()
    if current_client:
        print(f"Using custom client: {type(current_client).__name__}")
        if hasattr(current_client, 'base_url'):
            print(f"Base URL: {current_client.base_url}")
    else:
        print("Using default OpenAI client (from environment)")
    
    print("-" * 50)
    
    # --- Create and use agent ---
    agent = AssistantAgent()
    
    try:
        result = await agent.process(
            input="What's 2 + 2? Reply in one word."
        )
        print(f"Agent: {result['output']}")
        print(f"Tokens used: {result['tokens']['total_tokens']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you've configured a valid client or set OPENAI_API_KEY")
    
    finally:
        await shutdown()


# =============================================================================
# Production Pattern: FastAPI with Custom Client
# =============================================================================

"""
For production FastAPI apps, configure the client at startup:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from openai import AsyncOpenAI
from pyaiagent import set_default_openai_client, shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure client at startup
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        timeout=settings.OPENAI_TIMEOUT,
    )
    set_default_openai_client(client)
    
    # Create agent
    app.state.agent = MyAgent()
    
    yield
    
    # Cleanup
    await shutdown()

app = FastAPI(lifespan=lifespan)
```
"""


if __name__ == "__main__":
    asyncio.run(main())
