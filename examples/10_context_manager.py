"""
10_context_manager.py — Using Agents as Context Managers
=======================================================

pyaiagent agents support the async context manager protocol (async with).
This ensures proper cleanup and is especially useful in scripts,
tests, and one-off processing tasks.

What you'll learn:
  • Using agents with `async with`
  • When to use context managers vs. long-lived agents
  • Proper resource cleanup patterns
  • Parallel agent execution

Key concept:
  `async with` ensures aclose() is called automatically,
  even if an exception occurs. Great for scripts and testing!

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/10_context_manager.py
"""

import asyncio

from pyaiagent import OpenAIAgent, shutdown


# =============================================================================
# Define Agents
# =============================================================================

class AnalysisAgent(OpenAIAgent):
    """
    You are a data analyst.
    Analyze the provided data concisely.
    Focus on key insights and patterns.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.2


class CreativeAgent(OpenAIAgent):
    """
    You are a creative writer.
    Generate engaging, imaginative content.
    Be bold and original.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 1.0


class SummaryAgent(OpenAIAgent):
    """
    You are a summarization expert.
    Create clear, concise summaries.
    Capture the essential points only.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.1


# =============================================================================
# Example 1: Basic Context Manager Usage
# =============================================================================

async def example_basic_context_manager() -> None:
    """Basic async with pattern."""
    print("\n" + "=" * 60)
    print("📦 EXAMPLE 1: Basic Context Manager")
    print("=" * 60 + "\n")

    # The agent is automatically closed when the block exits
    async with AnalysisAgent() as agent:
        result = await agent.process(
            input="Analyze: Sales increased 25% in Q4, mainly in electronics."
        )
        print(f"Analysis: {result['output']}\n")

    # After the block, agent.aclose() has been called
    print("✅ Agent automatically closed after the block.")


# =============================================================================
# Example 2: Error Handling with Context Managers
# =============================================================================

async def example_error_handling() -> None:
    """Context managers ensure cleanup even on errors."""
    print("\n" + "=" * 60)
    print("⚠️  EXAMPLE 2: Cleanup on Error")
    print("=" * 60 + "\n")

    try:
        async with AnalysisAgent() as agent:
            # This will work
            result = await agent.process(input="Analyze: Revenue is up 10%")
            print(f"First call succeeded: {result['output'][:50]}...")

            # Simulate an application error
            raise ValueError("Simulated application error!")

    except ValueError as e:
        print(f"\n⚠️  Caught error: {e}")
        print("✅ Agent was still closed properly (via __aexit__)!")


# =============================================================================
# Example 3: Multiple Agents in Sequence
# =============================================================================

async def example_sequential_agents() -> None:
    """Use different agents for different stages of processing."""
    print("\n" + "=" * 60)
    print("🔗 EXAMPLE 3: Sequential Agent Pipeline")
    print("=" * 60 + "\n")

    data = """
    Q4 2024 Report:
    - Revenue: $2.4M (up 18%)
    - Customers: 1,240 (up 25%)
    - Churn: 3.2% (down from 4.1%)
    - Top product: Enterprise Plan (45% of revenue)
    - New market: APAC launched successfully
    """

    # Stage 1: Analyze
    print("Stage 1: Analysis...")
    async with AnalysisAgent() as analyst:
        analysis = await analyst.process(
            input=f"Analyze this business data:\n{data}"
        )
        print(f"   → {analysis['output'][:100]}...\n")

    # Stage 2: Summarize
    print("Stage 2: Summary...")
    async with SummaryAgent() as summarizer:
        summary = await summarizer.process(
            input=f"Summarize in 2 sentences: {analysis['output']}"
        )
        print(f"   → {summary['output']}\n")

    print("✅ Both agents cleaned up after their respective stages.")


# =============================================================================
# Example 4: Parallel Processing
# =============================================================================

async def example_parallel_processing() -> None:
    """Run multiple agents in parallel for speed."""
    print("\n" + "=" * 60)
    print("⚡ EXAMPLE 4: Parallel Agent Execution")
    print("=" * 60 + "\n")

    async def analyze_with_agent(agent_class, prompt: str) -> str:
        """Helper to run a single agent."""
        async with agent_class() as agent:
            result = await agent.process(input=prompt)
            return result["output"]

    # Run three different analyses in parallel
    print("Running 3 agents in parallel...\n")

    tasks = [
        analyze_with_agent(
            AnalysisAgent,
            "Analyze: User engagement increased 40% after the redesign."
        ),
        analyze_with_agent(
            CreativeAgent,
            "Write a one-sentence tagline for a productivity app."
        ),
        analyze_with_agent(
            SummaryAgent,
            "Summarize: Python is a popular language known for readability."
        ),
    ]

    results = await asyncio.gather(*tasks)

    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"  Agent {i}: {result[:80]}...")
    print()

    print("✅ All agents processed in parallel and cleaned up.")


# =============================================================================
# Example 5: Script Pattern (Full Cleanup)
# =============================================================================

async def example_script_pattern() -> None:
    """Recommended pattern for standalone scripts."""
    print("\n" + "=" * 60)
    print("📜 EXAMPLE 5: Complete Script Pattern")
    print("=" * 60 + "\n")

    try:
        # Multiple operations using context managers
        async with AnalysisAgent() as agent:
            result1 = await agent.process(input="What is 2 + 2?")
            result2 = await agent.process(input="What is 3 * 3?")
            print(f"Result 1: {result1['output']}")
            print(f"Result 2: {result2['output']}")

    finally:
        # Clean up the shared OpenAI client
        # This is important for scripts to avoid resource leaks!
        await shutdown()
        print("\n✅ shutdown() called - OpenAI client cleaned up.")


# =============================================================================
# When to Use Context Managers vs. Long-Lived Agents
# =============================================================================
#
# USE CONTEXT MANAGERS (async with):
#   ✓ Scripts and CLI tools
#   ✓ Testing
#   ✓ One-off processing tasks
#   ✓ Short-lived operations
#   ✓ Pipeline stages with different agents
#
# USE LONG-LIVED AGENTS (module-level instance):
#   ✓ Web servers (FastAPI, etc.)
#   ✓ Long-running services
#   ✓ When reusing the same agent repeatedly
#   ✓ High-throughput applications
#
# The key difference:
#   - Context managers: Create → Use → Close (for each block)
#   - Long-lived: Create once → Reuse many times → Close on shutdown


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    try:
        await example_basic_context_manager()
        await example_error_handling()
        await example_sequential_agents()
        await example_parallel_processing()

        # Run the script pattern example last (it calls shutdown)
        # await example_script_pattern()

        print("\n" + "=" * 60)
        print("✅ All context manager examples completed!")
        print("=" * 60)

    finally:
        # Always clean up the shared client in scripts
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Testing Pattern
# =============================================================================
#
# Context managers are great for pytest tests:
#
# import pytest
#
# @pytest.mark.asyncio
# async def test_agent_response():
#     async with MyAgent() as agent:
#         result = await agent.process(input="Test input")
#         assert "expected" in result["output"].lower()
#
# @pytest.fixture
# async def agent():
#     """Provide a fresh agent for each test."""
#     async with MyAgent() as agent:
#         yield agent
#     # Agent is automatically closed after test
#
# @pytest.mark.asyncio
# async def test_with_fixture(agent):
#     result = await agent.process(input="Hello")
#     assert result["output"]

