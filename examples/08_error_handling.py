"""
08_error_handling.py — Comprehensive Error Handling
==================================================

Production applications need robust error handling. pyaiagent provides
specific exception types for different error scenarios, making it easy
to handle each case appropriately.

What you'll learn:
  • All exception types pyaiagent can raise
  • How to catch specific errors vs. using the base class
  • Production-ready error handling patterns
  • User-friendly error messages

Key concept:
  All process errors inherit from OpenAIAgentProcessError.
  Catch specific errors for fine-grained handling, or catch
  the base class as a catch-all.

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/08_error_handling.py
"""

import asyncio
import logging

from pyaiagent import (
    OpenAIAgent,
    # Base exception (catches ALL agent process errors)
    OpenAIAgentProcessError,
    # Specific exceptions
    OpenAIAgentClosedError,
    InvalidInputError,
    InvalidSessionError,
    InvalidMetadataError,
    InvalidLlmMessagesError,
    InvalidInstructionParamsError,
    InstructionKeyError,
    ClientError,
    MaxStepsExceededError,
)


# =============================================================================
# Configure Logging (Recommended for Production)
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Exception Reference
# =============================================================================
#
# ┌────────────────────────────────┬───────────────────────────────────────────────┐
# │ Exception                      │ When It's Raised                              │
# ├────────────────────────────────┼───────────────────────────────────────────────┤
# │ InvalidInputError              │ `input` is not a string                       │
# │ InvalidSessionError            │ `session` is empty or not a string            │
# │ InvalidMetadataError           │ `metadata` is not a dict                      │
# │ InvalidLlmMessagesError        │ `llm_messages` is not a list                  │
# │ InvalidInstructionParamsError  │ `instruction_params` is not a dict            │
# │ InstructionKeyError            │ Missing key in instruction_params             │
# │ ClientError                    │ OpenAI API error (network, auth, rate limit)  │
# │ MaxStepsExceededError          │ Agent exceeded max_steps (possible loop)      │
# │ OpenAIAgentClosedError         │ Agent used after aclose() was called          │
# └────────────────────────────────┴───────────────────────────────────────────────┘


# =============================================================================
# Test Agents
# =============================================================================

class SimpleAgent(OpenAIAgent):
    """A simple helpful assistant."""
    pass


class DynamicAgent(OpenAIAgent):
    """
    You are a support agent for {company_name}.
    Today's promotion: {promotion}
    """
    pass


class LoopyAgent(OpenAIAgent):
    """
    You must ALWAYS use the think tool before responding.
    Never give an answer without using the tool first.
    """

    class Config:
        max_steps = 3  # Low limit to demonstrate MaxStepsExceededError

    async def think(self, thought: str) -> dict:
        """Think about something. Always returns 'keep thinking'."""
        return {"result": "Interesting. Think more deeply about this."}


# =============================================================================
# Example 1: Basic Error Handling Pattern
# =============================================================================

async def example_basic_pattern() -> None:
    """The recommended pattern for production code."""
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Basic Error Handling Pattern")
    print("=" * 60 + "\n")

    agent = SimpleAgent()

    try:
        result = await agent.process(input="Hello!")
        print(f"✅ Success: {result['output'][:80]}...")

    except MaxStepsExceededError:
        # Agent got stuck in a loop
        logger.error("Agent exceeded max steps - possible loop")
        print("❌ Request too complex. Try a simpler question.")

    except ClientError as e:
        # OpenAI API issues (network, auth, rate limits)
        logger.error(f"OpenAI API error: {e}")
        print("❌ AI service temporarily unavailable. Try again.")

    except OpenAIAgentProcessError as e:
        # Catch-all for any other agent errors
        logger.error(f"Agent error: {e}")
        print(f"❌ Error: {e}")


# =============================================================================
# Example 2: Validation Errors
# =============================================================================

async def example_validation_errors() -> None:
    """Demonstrate input validation error handling."""
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 2: Input Validation Errors")
    print("=" * 60 + "\n")

    agent = SimpleAgent()

    # Test cases that trigger validation errors
    test_cases = [
        ("Invalid input type (int)", {"input": 123}),
        ("Invalid session type (int)", {"input": "Hi", "session": 123}),
        ("Empty session string", {"input": "Hi", "session": "   "}),
        ("Invalid metadata type", {"input": "Hi", "metadata": "not-a-dict"}),
        ("Invalid llm_messages type", {"input": "Hi", "llm_messages": "not-a-list"}),
    ]

    for test_name, kwargs in test_cases:
        print(f"Testing: {test_name}...")
        try:
            await agent.process(**kwargs)  # type: ignore
            print(f"   ⚠️ No error raised (unexpected)\n")
        except (
            InvalidInputError,
            InvalidSessionError,
            InvalidMetadataError,
            InvalidLlmMessagesError
        ) as e:
            print(f"   ✅ Caught: {type(e).__name__}\n")


# =============================================================================
# Example 3: Dynamic Instruction Errors
# =============================================================================

async def example_instruction_errors() -> None:
    """Demonstrate instruction parameter errors."""
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 3: Dynamic Instruction Errors")
    print("=" * 60 + "\n")

    agent = DynamicAgent()

    # Missing required placeholders
    print("Test 1: Missing instruction_params...")
    try:
        await agent.process(input="What's the promotion?")
    except InstructionKeyError as e:
        print(f"   ✅ Caught InstructionKeyError: Missing key\n")

    # Wrong type for instruction_params
    print("Test 2: Wrong type for instruction_params...")
    try:
        await agent.process(
            input="Hi",
            instruction_params="not-a-dict"  # type: ignore
        )
    except InvalidInstructionParamsError:
        print(f"   ✅ Caught InvalidInstructionParamsError\n")

    # Correct usage
    print("Test 3: Correct usage (all params provided)...")
    try:
        result = await agent.process(
            input="What's the promotion?",
            instruction_params={
                "company_name": "TechCorp",
                "promotion": "50% off all plans!"
            }
        )
        print(f"   ✅ Success: {result['output'][:60]}...\n")
    except OpenAIAgentProcessError as e:
        print(f"   ❌ Error: {e}\n")


# =============================================================================
# Example 4: Max Steps Exceeded (Loop Detection)
# =============================================================================

async def example_max_steps() -> None:
    """Demonstrate max_steps protection against infinite loops."""
    print("\n" + "=" * 60)
    print("🔄 EXAMPLE 4: Max Steps Exceeded (Loop Protection)")
    print("=" * 60 + "\n")

    agent = LoopyAgent()  # Has max_steps=3

    print("Running agent with a tool that causes loops...")
    print("(max_steps=3 will trigger the error)\n")

    try:
        await agent.process(
            input="Analyze the meaning of life in great detail."
        )
    except MaxStepsExceededError as e:
        print(f"✅ Caught MaxStepsExceededError!")
        print(f"   The agent was stopped after {agent._config.max_steps} steps.")
        print(f"   This prevents runaway costs and infinite loops.\n")


# =============================================================================
# Example 5: Closed Agent Error
# =============================================================================

async def example_closed_agent() -> None:
    """Demonstrate error when using a closed agent."""
    print("\n" + "=" * 60)
    print("🚫 EXAMPLE 5: Closed Agent Error")
    print("=" * 60 + "\n")

    agent = SimpleAgent()

    # First call works
    print("Step 1: Normal usage (agent is open)...")
    result = await agent.process(input="Hello!")
    print(f"   ✅ Success: {result['output'][:40]}...\n")

    # Close the agent
    print("Step 2: Closing the agent with aclose()...")
    await agent.aclose()
    print("   ✅ Agent closed.\n")

    # Try to use it after closing
    print("Step 3: Attempting to use closed agent...")
    try:
        await agent.process(input="Hello again!")
    except OpenAIAgentClosedError:
        print("   ✅ Caught OpenAIAgentClosedError!")
        print("   💡 Solution: Create a new agent instance.\n")


# =============================================================================
# Example 6: Production-Ready Error Handler
# =============================================================================

async def safe_process(
    agent: OpenAIAgent,
    user_input: str,
    **kwargs
) -> dict:
    """
    Production-ready wrapper with comprehensive error handling.

    Returns a consistent response structure with:
    - success: bool
    - response: str (if success)
    - error: str (if failure)
    - error_code: str (if failure, for programmatic handling)
    """
    try:
        result = await agent.process(input=user_input, **kwargs)
        return {
            "success": True,
            "response": result["output"],
            "tokens": result["tokens"]["total_tokens"],
            "session": result["session"]
        }

    except InvalidInputError:
        return {
            "success": False,
            "error": "Please provide a valid text message.",
            "error_code": "INVALID_INPUT"
        }

    except InvalidSessionError:
        return {
            "success": False,
            "error": "Invalid session ID provided.",
            "error_code": "INVALID_SESSION"
        }

    except InstructionKeyError as e:
        logger.error(f"Missing instruction param: {e}")
        return {
            "success": False,
            "error": "Configuration error. Please contact support.",
            "error_code": "CONFIG_ERROR"
        }

    except MaxStepsExceededError:
        logger.warning(f"Max steps exceeded for: {user_input[:50]}...")
        return {
            "success": False,
            "error": "Request too complex. Try breaking it into smaller questions.",
            "error_code": "TOO_COMPLEX"
        }

    except ClientError as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "success": False,
            "error": "AI service temporarily unavailable. Please retry.",
            "error_code": "SERVICE_ERROR"
        }

    except OpenAIAgentClosedError:
        logger.critical("Attempted to use closed agent!")
        return {
            "success": False,
            "error": "Internal error. Please try again.",
            "error_code": "INTERNAL_ERROR"
        }

    except OpenAIAgentProcessError as e:
        logger.error(f"Unexpected agent error: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred.",
            "error_code": "UNKNOWN_ERROR"
        }


async def example_production_handler() -> None:
    """Demonstrate the production-ready error handler."""
    print("\n" + "=" * 60)
    print("🏭 EXAMPLE 6: Production-Ready Handler")
    print("=" * 60 + "\n")

    agent = SimpleAgent()

    # Test with valid input
    print("Test 1: Valid input...")
    response = await safe_process(agent, "What is Python?")
    print(f"   Result: success={response['success']}")
    if response["success"]:
        print(f"   Response: {response['response'][:60]}...\n")

    # Test with invalid input
    print("Test 2: Invalid input (integer)...")
    response = await safe_process(agent, 12345)  # type: ignore
    print(f"   Result: success={response['success']}")
    print(f"   Error: {response.get('error')}")
    print(f"   Code: {response.get('error_code')}\n")


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    await example_basic_pattern()
    await example_validation_errors()
    await example_instruction_errors()
    await example_max_steps()
    await example_closed_agent()
    await example_production_handler()

    print("\n" + "=" * 60)
    print("✅ All error handling examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# FastAPI Error Handling Example
# =============================================================================
#
# For FastAPI, map error codes to HTTP status codes:
#
# from fastapi import HTTPException
#
# async def chat_endpoint(message: str):
#     response = await safe_process(agent, message)
#
#     if not response["success"]:
#         status_map = {
#             "INVALID_INPUT": 400,
#             "INVALID_SESSION": 400,
#             "CONFIG_ERROR": 500,
#             "TOO_COMPLEX": 422,
#             "SERVICE_ERROR": 503,
#             "INTERNAL_ERROR": 500,
#             "UNKNOWN_ERROR": 500,
#         }
#         status = status_map.get(response["error_code"], 500)
#         raise HTTPException(status_code=status, detail=response["error"])
#
#     return {"response": response["response"]}

