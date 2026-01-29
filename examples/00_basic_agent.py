"""
00_basic_agent.py — Your First AI Agent
========================================

This is the simplest possible example of using pyaiagent.
Perfect for getting started in under 2 minutes!

What you'll learn:
  • How to create an AI agent by subclassing OpenAIAgent
  • How the class docstring becomes the agent's "instructions"
  • How to run the agent and get a response

Prerequisites:
  1. Install the package:     pip install pyaiagent
  2. Set your API key:        export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/00_basic_agent.py

Expected output:
  The agent will respond warmly to your greeting!
"""

import asyncio

from pyaiagent import OpenAIAgent


# =============================================================================
# STEP 1: Define Your Agent
# =============================================================================
#
# To create an agent, simply subclass OpenAIAgent and write a docstring.
# The docstring becomes the "system instructions" — it tells the AI:
#   • What persona to take
#   • How to behave
#   • Any rules to follow
#
# That's it! No decorators, no config files, no magic.

class FriendlyAssistant(OpenAIAgent):
    """
    You are a friendly and helpful assistant.
    You always respond in a warm, encouraging tone.
    Keep your answers concise but informative.
    """


# =============================================================================
# STEP 2: Run the Agent
# =============================================================================
#
# pyaiagent is async-first, so we use `async def` and `asyncio.run()`.
# If you're using FastAPI, you're already in an async context!

async def main() -> None:
    # Create an instance of your agent.
    # The agent is lightweight — it only connects to OpenAI when you call process().
    agent = FriendlyAssistant()

    # Send a message to the agent.
    # The `input` parameter is the user's message (always required).
    result = await agent.process(
        input="Hello! Is building an AI agent really this simple?"
    )

    # ==========================================================================
    # Understanding the Result
    # ==========================================================================
    #
    # The result is a dictionary with several useful fields:
    #
    #   result["output"]              → The agent's text response (string)
    #   result["tokens"]              → Token usage for cost tracking
    #   result["messages"]["llm"]     → Conversation history (for multi-turn)
    #   result["session"]             → Session ID (auto-generated or provided)
    #   result["steps"]               → How many LLM calls were made
    #
    # For now, we just need the output:

    print("─" * 60)
    print("🤖 Agent's Response:")
    print("─" * 60)
    print(result["output"])
    print("─" * 60)

    # Bonus: Print token usage (useful for monitoring costs)
    tokens = result["tokens"]
    print(
        f"\n📊 Tokens used: {tokens['total_tokens']} total "
        f"({tokens['input_tokens']} input + {tokens['output_tokens']} output)"
    )


# =============================================================================
# Entry Point
# =============================================================================
#
# This pattern ensures the code only runs when executed directly,
# not when imported as a module.

if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# What's Next?
# =============================================================================
#
# Now that you've created your first agent, explore these concepts:
#
#   01_conversation_memory.py  → Multi-turn conversations
#   02_tools_basic.py          → Give your agent superpowers with tools
#   03_structured_output.py    → Get typed responses with Pydantic
#   04_config_advanced.py      → Customize model, temperature, timeouts
#
# Happy building! 🚀
