"""
04_config_advanced.py — Advanced Configuration
==============================================

The nested `Config` class lets you customize every aspect of your agent's
behavior — from model selection to timeouts to tool execution.

What you'll learn:
  • All available configuration options
  • When and why to change each setting
  • Patterns for different use cases (creative, precise, fast)

Prerequisites:
  pip install pyaiagent pydantic
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/04_config_advanced.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from pyaiagent import OpenAIAgent


# =============================================================================
# Configuration Reference
# =============================================================================
#
# ┌─────────────────────────┬──────────────┬───────────────────────────────────────────┐
# │ Option                  │ Default      │ Description                               │
# ├─────────────────────────┼──────────────┼───────────────────────────────────────────┤
# │ model                   │ "gpt-4o-mini"│ OpenAI model to use                       │
# │ temperature             │ 0.2          │ Randomness (0.0-2.0, lower = deterministic│
# │ top_p                   │ None         │ Nucleus sampling (alternative to temp)    │
# │ max_output_tokens       │ 4096         │ Maximum tokens in response                │
# │ seed                    │ None         │ For reproducible outputs                  │
# ├─────────────────────────┼──────────────┼───────────────────────────────────────────┤
# │ tool_choice             │ "auto"       │ "auto", "none", or "required"             │
# │ parallel_tool_calls     │ True         │ Allow multiple simultaneous tool calls    │
# │ max_steps               │ 10           │ Max LLM ↔ tool rounds per request         │
# │ max_parallel_tools      │ 10           │ Concurrency limit for tool execution      │
# │ tool_timeout            │ 30.0         │ Timeout per tool call (seconds)           │
# │ llm_timeout             │ 120.0        │ Timeout for LLM response (seconds)        │
# ├─────────────────────────┼──────────────┼───────────────────────────────────────────┤
# │ text_format             │ None         │ Pydantic model for structured output      │
# │ include_events          │ True         │ Include event records for UI/DB/analytics  │
# │ include_history    │ True         │ Include LLM messages for memory           │
# └─────────────────────────┴──────────────┴───────────────────────────────────────────┘


# =============================================================================
# Example 1: Creative Writing (High Temperature)
# =============================================================================

class CreativeWriter(OpenAIAgent):
    """
    You are an award-winning creative writer with a vivid imagination.
    Write engaging, original content with flair and unexpected twists.
    Don't be afraid to be unconventional or poetic.
    """

    class Config:
        model = "gpt-4o"           # More capable model for creative tasks
        temperature = 1.2          # Higher = more creative and varied
        max_output_tokens = 2048   # Allow longer creative pieces


# =============================================================================
# Example 2: Code Assistant (Low Temperature, Reproducible)
# =============================================================================

class CodeAssistant(OpenAIAgent):
    """
    You are a precise, senior software engineer.
    Write clean, correct, well-documented code.
    Follow best practices and explain your reasoning.
    """

    class Config:
        model = "gpt-4o"
        temperature = 0.1          # Very low = consistent, predictable
        seed = 42                  # Same input → same output (reproducibility)
        max_output_tokens = 4096   # Allow detailed code explanations


# =============================================================================
# Example 3: Fast Classifier (Optimized for Speed)
# =============================================================================

class SentimentLabel(BaseModel):
    """Simple sentiment classification."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, description="0.0 to 1.0")


class QuickClassifier(OpenAIAgent):
    """
    Classify the sentiment of the given text.
    Be accurate and confident in your classification.
    """

    class Config:
        model = "gpt-4o-mini"      # Fastest model
        temperature = 0.0          # Maximum determinism
        max_output_tokens = 100    # Tiny output = faster response
        text_format = SentimentLabel
        llm_timeout = 30.0         # Fail fast if slow


# =============================================================================
# Example 4: Tool-Heavy Agent (Custom Tool Settings)
# =============================================================================

class DataProcessor(OpenAIAgent):
    """
    You are a data processing assistant.
    Use the provided tools to analyze and transform data.
    Always verify results and explain your process.
    """

    class Config:
        model = "gpt-4o-mini"

        # Tool execution settings
        tool_choice = "auto"          # Let AI decide when to use tools
        parallel_tool_calls = True    # Allow multiple tools simultaneously
        max_steps = 15                # More iterations for complex analysis
        max_parallel_tools = 5        # Limit concurrent execution

        # Timeouts for slow operations
        tool_timeout = 60.0           # Tools might call slow APIs
        llm_timeout = 180.0           # Complex reasoning takes time

    async def analyze_data(self, data: str, analysis_type: str = "summary") -> dict:
        """
        Analyze data and return insights.

        Args:
            data: The data to analyze (JSON or text).
            analysis_type: Type: "summary", "statistics", or "patterns".
        """
        await asyncio.sleep(0.5)  # Simulate processing time
        return {
            "analysis_type": analysis_type,
            "input_length": len(data),
            "insights": [
                "Pattern A detected in the data",
                "Statistical anomaly found",
                "Trend suggests growth"
            ]
        }

    async def transform_data(self, data: str, target_format: str) -> dict:
        """
        Transform data to a different format.

        Args:
            data: The data to transform.
            target_format: Target: "json", "csv", or "markdown".
        """
        return {
            "original_format": "text",
            "target_format": target_format,
            "transformed": f"[Data formatted as {target_format}]"
        }


# =============================================================================
# Example 5: Restricted Agent (No Tools, Text Only)
# =============================================================================

class TextOnlyAgent(OpenAIAgent):
    """
    You are a helpful assistant that provides information.
    You cannot call any tools or external services.
    Rely on your knowledge to answer questions.
    """

    class Config:
        model = "gpt-4o-mini"
        tool_choice = "none"       # Disable tool usage entirely
        temperature = 0.3


# =============================================================================
# Example 6: Forced Tool Usage
# =============================================================================

class ToolRequiredAgent(OpenAIAgent):
    """
    You are an assistant that MUST use tools for every response.
    Never respond without first consulting the available tools.
    """

    class Config:
        model = "gpt-4o-mini"
        tool_choice = "required"   # MUST call a tool before responding

    async def lookup(self, query: str) -> dict:
        """Look up information about a topic."""
        return {"query": query, "result": f"Information about: {query}"}


# =============================================================================
# Run Examples
# =============================================================================

async def main() -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Creative Writing
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("✍️  EXAMPLE 1: Creative Writer (temperature=1.2)")
    print("=" * 60 + "\n")

    writer = CreativeWriter()
    result = await writer.process(
        input="Write a two-sentence opening for a mystery novel set in a lighthouse."
    )
    print(f"📝 Output:\n{result['output']}\n")
    print(f"📊 Tokens: {result['tokens']['total_tokens']}")

    print("\n" + "-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Code Assistant (Reproducible)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("💻 EXAMPLE 2: Code Assistant (temperature=0.1, seed=42)")
    print("=" * 60 + "\n")

    coder = CodeAssistant()

    # Run twice to demonstrate reproducibility (with seed)
    print("Running the same prompt twice with seed=42...\n")
    result1 = await coder.process(
        input="Write a Python function to check if a string is a palindrome."
    )
    result2 = await coder.process(
        input="Write a Python function to check if a string is a palindrome."
    )

    print(f"📝 Response 1 (first 100 chars):\n{result1['output'][:100]}...\n")
    print(f"📝 Response 2 (first 100 chars):\n{result2['output'][:100]}...")
    print(f"\n💡 With seed=42, outputs should be identical (or very similar).")

    print("\n" + "-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Quick Classification
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("⚡ EXAMPLE 3: Quick Classifier (optimized for speed)")
    print("=" * 60 + "\n")

    classifier = QuickClassifier()

    samples = [
        "I love this product! Best purchase ever!",
        "Meh, it's okay I guess.",
        "Worst experience of my life. Completely broken.",
    ]

    for text in samples:
        result = await classifier.process(input=f"Classify: {text}")
        label: SentimentLabel = result["output_parsed"]
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}[label.sentiment]
        print(f'"{text[:40]}..."')
        print(f"   {emoji} {label.sentiment} ({label.confidence:.0%} confident)\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 4: Data Processor (Tool-Heavy)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🔧 EXAMPLE 4: Data Processor (max_steps=15, tool_timeout=60s)")
    print("=" * 60 + "\n")

    processor = DataProcessor()
    result = await processor.process(
        input="Analyze this sales data and summarize the patterns: "
              "[{'product': 'A', 'sales': 150}, {'product': 'B', 'sales': 230}]"
    )

    print(f"📝 Analysis:\n{result['output']}")
    print(f"\n📊 Steps taken: {result['steps']}")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Configuration Best Practices
# =============================================================================
#
# 1. START WITH DEFAULTS
#    The defaults work well for most use cases. Only change what you need.
#
# 2. TEMPERATURE GUIDELINES
#    • 0.0-0.3: Classification, code, factual Q&A
#    • 0.3-0.7: General conversation, balanced responses
#    • 0.7-1.2: Creative writing, brainstorming
#    • 1.2+:    Highly creative (may be unpredictable)
#
# 3. MODEL SELECTION
#    • gpt-4o-mini: Fast, cheap, good for most tasks
#    • gpt-4o: More capable, better reasoning, higher cost
#    • Check OpenAI docs for latest models
#
# 4. PROTECT AGAINST LOOPS
#    Always set a reasonable `max_steps` (default is 10).
#    This prevents runaway tool loops.
#
# 5. SET REALISTIC TIMEOUTS
#    • tool_timeout: Match your slowest external API
#    • llm_timeout: Increase for complex reasoning tasks
#
# 6. USE SEED FOR TESTING
#    Set seed=N for reproducible outputs during development/testing.
#    Note: Not 100% guaranteed, but greatly improves consistency.
