"""
01_conversation_memory.py — Multi-Turn Conversations
====================================================

By default, each process() call is independent — the agent has no memory.
This example shows how to enable conversation history so the agent
"remembers" what was said in previous messages.

What you'll learn:
  • How to pass conversation history using `history`
  • How the agent maintains context across multiple turns
  • Patterns for building chat applications

Key concept:
  Pass result["history"] from one call into the next call's
  `history` parameter. That's all it takes!

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/01_conversation_memory.py
"""

import asyncio
from typing import Any

from pyaiagent import OpenAIAgent


# =============================================================================
# Define the Agent
# =============================================================================

class ConversationalAssistant(OpenAIAgent):
    """
    You are a friendly conversational assistant with excellent memory.
    You remember everything the user tells you within the conversation.
    When the user asks about something they mentioned earlier, recall it accurately.
    Be warm and personable in your responses.
    """


# =============================================================================
# Example 1: Basic Multi-Turn Conversation
# =============================================================================

async def example_basic_memory() -> None:
    """
    Demonstrate the simplest pattern for conversation memory.
    Pass the previous result's messages to the next call.
    """
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 1: Basic Multi-Turn Memory")
    print("=" * 60 + "\n")

    agent = ConversationalAssistant()

    # ─────────────────────────────────────────────────────────────────────────
    # Turn 1: User introduces themselves
    # ─────────────────────────────────────────────────────────────────────────
    print("You: My name is Alex and I'm a software engineer who loves hiking.")
    result1 = await agent.process(
        input="My name is Alex and I'm a software engineer who loves hiking."
    )
    print(f"Agent: {result1['output']}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Turn 2: Test the agent's memory
    # ─────────────────────────────────────────────────────────────────────────
    # KEY: Pass the previous messages to maintain context!
    print("You: What's my name and what are my hobbies?")
    result2 = await agent.process(
        input="What's my name and what are my hobbies?",
        history=result1["history"]  # ← This enables memory!
    )
    print(f"Agent: {result2['output']}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Turn 3: Continue the conversation
    # ─────────────────────────────────────────────────────────────────────────
    # Always use the LATEST result's messages
    print("You: Can you recommend a hiking trail for someone in my profession?")
    result3 = await agent.process(
        input="Can you recommend a hiking trail for someone in my profession?",
        history=result2["history"]  # ← Updated messages!
    )
    print(f"Agent: {result3['output']}\n")


# =============================================================================
# Example 2: Chat Loop Pattern
# =============================================================================

async def example_chat_loop() -> None:
    """
    The recommended pattern for building a chat application.
    Maintain a single `history` variable and update it after each turn.
    """
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 2: Chat Loop Pattern (Programmatic)")
    print("=" * 60 + "\n")

    agent = ConversationalAssistant()

    # A scripted conversation to demonstrate the pattern
    user_messages = [
        "Hi! I'm planning a trip to Japan next month.",
        "What's the best time to see cherry blossoms?",
        "Which city should I visit first?",
        "Wait, what was I planning again?",  # Tests if agent remembers!
    ]

    # Start with an empty list — no previous messages
    history: list[dict[str, Any]] = []

    for user_message in user_messages:
        print(f"You: {user_message}")

        result = await agent.process(
            input=user_message,
            history=history  # Pass current history
        )

        # IMPORTANT: Update history with the NEW conversation state
        # This includes the user message we just sent + the agent's response
        history = result["history"]

        print(f"Agent: {result['output']}\n")
        print("─" * 40 + "\n")

    # Show final message count
    print(f"📊 Total messages in conversation: {len(history)}")


# =============================================================================
# Example 3: Interactive Chat (Try it yourself!)
# =============================================================================

async def example_interactive_chat() -> None:
    """
    An interactive chat where you can test the memory yourself.
    Tell the agent your name, then ask what your name is!
    """
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 3: Interactive Chat")
    print("=" * 60)
    print("Chat with the agent! It will remember what you say.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    agent = ConversationalAssistant()
    history: list[dict[str, Any]] = []

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        # Handle empty input or exit commands
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        # Process the message with conversation history
        result = await agent.process(
            input=user_input,
            history=history
        )

        # Update the conversation history
        history = result["history"]

        print(f"Agent: {result['output']}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main() -> None:
    # Run the scripted examples
    await example_basic_memory()
    await example_chat_loop()

    # Uncomment below to try the interactive chat:
    # await example_interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Tips for Production
# =============================================================================
#
# 1. STORE MESSAGES EXTERNALLY
#    For real apps, store history in Redis, a database, or session storage.
#    The in-memory approach shown here is for demonstration only.
#
# 2. LIMIT CONVERSATION LENGTH
#    Long conversations use more tokens. Consider truncating old messages:
#
#      MAX_MESSAGES = 20
#      if len(history) > MAX_MESSAGES:
#          history = history[-MAX_MESSAGES:]
#
# 3. USE SESSION IDs
#    Track conversations with the optional `session` parameter:
#
#      result = await agent.process(
#          input=user_message,
#          session="user-123-conversation-456",
#          history=history
#      )
#
# 4. SEE THE FASTAPI EXAMPLE
#    Check out 06_fastapi_with_lifespan.py for a production-ready
#    implementation with proper session management.
