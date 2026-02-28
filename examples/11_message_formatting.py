"""
11_message_formatting.py — Optimizing Token Usage with Message Hooks
====================================================================

When using structured outputs with large fields, conversation memory
can explode in token usage. The message formatting hooks let you control
what gets stored — saving significant costs!

What you'll learn:
  • How `format_history()` customizes LLM conversation memory
  • How `format_event()` customizes UI/frontend display
  • Token optimization strategies for structured outputs

Key concept:
  Override `format_history()` to store only essential content in
  conversation history. The large data stays in your app, not in every
  API call to OpenAI.

Prerequisites:
  pip install pyaiagent pydantic
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/11_message_formatting.py
"""

import asyncio

from pydantic import BaseModel, Field

from pyaiagent import OpenAIAgent


# =============================================================================
# Example 1: The Problem — Large Structured Output
# =============================================================================
#
# Imagine an agent that returns both a short response AND large data.
# Without optimization, the entire output is stored in conversation memory,
# causing token costs to compound with every turn.

class ReportOutput(BaseModel):
    """Agent output with a small response and large data."""
    
    summary: str = Field(description="Brief summary for the user (1-2 sentences)")
    detailed_report: str = Field(description="Detailed analysis (can be very long)")
    recommendations: list[str] = Field(description="List of recommendations")


class UnoptimizedReportAgent(OpenAIAgent):
    """
    You are a business analyst. Provide a brief summary and detailed report.
    Make the detailed_report comprehensive (at least 500 words).
    """

    class Config:
        model = "gpt-4o-mini"
        text_format = ReportOutput


# =============================================================================
# Example 2: The Solution — Optimized Message Formatting
# =============================================================================
#
# By overriding format_history(), we store only the summary in conversation
# memory. The detailed_report is still returned to the caller but doesn't
# bloat the conversation history.

class OptimizedReportAgent(OpenAIAgent):
    """
    You are a business analyst. Provide a brief summary and detailed report.
    Make the detailed_report comprehensive (at least 500 words).
    """

    class Config:
        model = "gpt-4o-mini"
        text_format = ReportOutput

    def format_history(self, response) -> str:
        """
        Only store the summary in conversation memory.
        This dramatically reduces token usage for multi-turn conversations!
        """
        if response.output_parsed:
            return f"Summary: {response.output_parsed.summary}"
        return response.output_text or ""

    def format_event(self, response) -> str:
        """
        For UI, we might want to show a bit more context.
        """
        if response.output_parsed:
            parsed = response.output_parsed
            return f"Summary: {parsed.summary}\n\nRecommendations: {', '.join(parsed.recommendations)}"
        return response.output_text or ""


# =============================================================================
# Example 3: Chat Bot with Context — Real-World Pattern
# =============================================================================
#
# A chatbot that returns structured responses with metadata.
# Only the user-facing message should be in conversation memory.

class ChatResponse(BaseModel):
    """Structured chat response with metadata."""
    
    message: str = Field(description="The response message to show the user")
    intent: str = Field(description="Detected user intent")
    confidence: float = Field(description="Confidence score 0-1")
    suggested_actions: list[str] = Field(description="Suggested follow-up actions")
    internal_notes: str = Field(description="Internal notes for logging (not for user)")


class SmartChatBot(OpenAIAgent):
    """
    You are a helpful customer service chatbot.
    Classify user intent and provide helpful responses.
    Include internal notes for analytics purposes.
    """

    class Config:
        model = "gpt-4o-mini"
        text_format = ChatResponse

    def format_history(self, response) -> str:
        """
        Only store the user-facing message in conversation history.
        Intent, confidence, and internal_notes are metadata — don't need them in memory.
        """
        if response.output_parsed:
            return response.output_parsed.message
        return response.output_text or ""

    def format_event(self, response) -> str:
        """
        For UI/logging, include the message and suggested actions.
        """
        if response.output_parsed:
            parsed = response.output_parsed
            actions = ", ".join(parsed.suggested_actions) if parsed.suggested_actions else "None"
            return f"{parsed.message}\n\n[Suggested: {actions}]"
        return response.output_text or ""


# =============================================================================
# Example 4: Comparing Token Usage
# =============================================================================

async def compare_token_usage():
    """Demonstrate the token savings from message formatting."""
    
    print("=" * 60)
    print("📊 Token Usage Comparison")
    print("=" * 60)
    
    # Simulate a multi-turn conversation
    unoptimized = UnoptimizedReportAgent()
    optimized = OptimizedReportAgent()
    
    question = "Analyze the impact of remote work on productivity."
    
    # First turn
    print("\n--- Turn 1 ---")
    
    result_unopt = await unoptimized.process(input=question)
    result_opt = await optimized.process(input=question)
    
    print(f"Unoptimized - Tokens: {result_unopt['tokens']['total_tokens']}")
    print(f"Optimized   - Tokens: {result_opt['tokens']['total_tokens']}")
    
    # Check what's stored in history
    unopt_msg_len = len(str(result_unopt['history']))
    opt_msg_len = len(str(result_opt['history']))
    
    print(f"\nMessage storage size:")
    print(f"Unoptimized - {unopt_msg_len} chars")
    print(f"Optimized   - {opt_msg_len} chars")
    print(f"Savings     - {100 - (opt_msg_len / unopt_msg_len * 100):.1f}%")
    
    # Second turn (shows compounding effect)
    print("\n--- Turn 2 (with conversation history) ---")
    
    followup = "What are the top 3 recommendations?"
    
    result_unopt2 = await unoptimized.process(
        input=followup, 
        history=result_unopt['history']
    )
    result_opt2 = await optimized.process(
        input=followup, 
        history=result_opt['history']
    )
    
    print(f"Unoptimized - Turn 2 Tokens: {result_unopt2['tokens']['total_tokens']}")
    print(f"Optimized   - Turn 2 Tokens: {result_opt2['tokens']['total_tokens']}")
    
    savings = result_unopt2['tokens']['total_tokens'] - result_opt2['tokens']['total_tokens']
    print(f"Token Savings: {savings} tokens")


# =============================================================================
# Example 5: Chat Bot Demo
# =============================================================================

async def chatbot_demo():
    """Interactive demo of the smart chatbot."""
    
    print("\n" + "=" * 60)
    print("🤖 Smart ChatBot Demo")
    print("=" * 60)
    
    bot = SmartChatBot()
    history = None
    
    conversations = [
        "Hi, I need help with my order",
        "It hasn't arrived yet, order #12345",
        "Can you check the shipping status?"
    ]
    
    for user_input in conversations:
        print(f"\n👤 User: {user_input}")
        
        result = await bot.process(input=user_input, history=history)
        history = result['history']
        
        # Show the structured response
        parsed = result['output_parsed']
        print(f"🤖 Bot: {parsed.message}")
        print(f"   Intent: {parsed.intent} (confidence: {parsed.confidence:.0%})")
        print(f"   Suggestions: {', '.join(parsed.suggested_actions)}")
        
        # Show what's stored in memory (lean!)
        last_assistant_msg = [m for m in history if m.get('role') == 'assistant'][-1]
        print(f"   [Memory stores: \"{last_assistant_msg['content'][:50]}...\"]")


# =============================================================================
# Run Examples
# =============================================================================

async def main():
    await compare_token_usage()
    await chatbot_demo()
    
    print("\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("=" * 60)
    print("""
Key Takeaways:
1. Override format_history() to reduce token usage
2. Only store what's needed for conversation context
3. Token savings compound with each turn
4. format_event() can be different for display purposes
""")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Message Formatting Best Practices
# =============================================================================
#
# 1. IDENTIFY WHAT'S NEEDED FOR CONTEXT
#    Ask: "What does the AI need to remember to continue the conversation?"
#    Usually just the user-facing message, not metadata or large data.
#
# 2. KEEP LARGE DATA SEPARATE
#    Return large data in structured output, but don't put it in memory.
#    Store it in your database or return it to your frontend separately.
#
# 3. DIFFERENT HOOKS FOR DIFFERENT PURPOSES
#    - format_history(): What the AI needs for context (optimize for tokens)
#    - format_event(): What to show users/logs (optimize for clarity)
#
# 4. TEST YOUR TOKEN USAGE
#    Compare total_tokens across turns with and without optimization.
#    The savings compound significantly in long conversations.
#
# 5. WHEN NOT TO OPTIMIZE
#    If your structured output is small (< 100 tokens), optimization
#    may not be worth the added complexity.
