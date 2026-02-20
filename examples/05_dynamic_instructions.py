"""
05_dynamic_instructions.py — Personalized Agents at Runtime
==========================================================

Sometimes you need to customize the agent's instructions for each request —
user preferences, current date, or context-specific rules. Dynamic instructions
let you do this with simple {placeholders}.

What you'll learn:
  • How to use {placeholders} in agent docstrings
  • How to pass values via `instruction_params`
  • Patterns for personalization and context injection

Key concept:
  Add {placeholder_name} in your docstring, then pass values via
  instruction_params={"placeholder_name": "value"} in process().

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/05_dynamic_instructions.py
"""

import asyncio
from datetime import datetime

from pyaiagent import OpenAIAgent


# =============================================================================
# Example 1: Personalized Assistant
# =============================================================================

class PersonalizedAssistant(OpenAIAgent):
    """
    You are a personal assistant for {user_name}.

    About {user_name}:
    - Preferences: {preferences}
    - Communication style: {style}

    Today's date is {date}.
    Always address them by name and respect their preferences.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.5


# =============================================================================
# Example 2: Customer Support with Company Context
# =============================================================================

class CustomerSupportAgent(OpenAIAgent):
    """
    You are a customer support agent for {company_name}.

    Company information:
    - Industry: {industry}
    - Support hours: {support_hours}
    - Current promotion: {promotion}

    Always be professional and helpful. Reference the current promotion
    when relevant. If issues can't be resolved, direct customers to
    contact support during {support_hours}.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.3


# =============================================================================
# Example 3: Multi-Language Support
# =============================================================================

class MultiLanguageAgent(OpenAIAgent):
    """
    You are a helpful assistant.

    IMPORTANT: You MUST respond in {language}.
    The user's preferred name is {user_name}.
    Use a {tone} tone in your responses.

    If you don't know something, say so in {language}.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.4


# =============================================================================
# Example 4: Role-Based Agent
# =============================================================================

class RoleBasedAgent(OpenAIAgent):
    """
    You are playing the role of: {role}

    Character details:
    {character_description}

    Stay in character for all responses. React as your character would.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.7  # More creative for roleplay


# =============================================================================
# Run Examples
# =============================================================================

async def main() -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Personalized Assistant
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("👤 EXAMPLE 1: Personalized Assistant")
    print("=" * 60 + "\n")

    assistant = PersonalizedAssistant()

    # User A: Casual tech enthusiast
    print("--- User A: Alex (casual, tech enthusiast) ---\n")
    result = await assistant.process(
        input="What should I do this weekend?",
        instruction_params={
            "user_name": "Alex",
            "preferences": "loves technology, coffee, and hiking",
            "style": "casual and friendly",
            "date": datetime.now().strftime("%A, %B %d, %Y")
        }
    )
    print(f"Question: What should I do this weekend?\n")
    print(f"Agent: {result['output']}\n")

    # User B: Professional finance person
    print("--- User B: Victoria (formal, finance professional) ---\n")
    result = await assistant.process(
        input="What should I prioritize today?",
        instruction_params={
            "user_name": "Victoria",
            "preferences": "interested in financial markets, values efficiency",
            "style": "formal and concise",
            "date": datetime.now().strftime("%A, %B %d, %Y")
        }
    )
    print(f"Question: What should I prioritize today?\n")
    print(f"Agent: {result['output']}\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Customer Support with Company Context
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🏢 EXAMPLE 2: Customer Support (Company-Specific)")
    print("=" * 60 + "\n")

    support = CustomerSupportAgent()

    # TechCorp configuration
    result = await support.process(
        input="Do you have any deals right now? I'm thinking of upgrading.",
        instruction_params={
            "company_name": "TechCorp Software",
            "industry": "SaaS / Cloud Software",
            "support_hours": "Monday-Friday, 9 AM - 6 PM EST",
            "promotion": "50% off annual plans until end of month!"
        }
    )

    print(f"Customer: Do you have any deals right now?\n")
    print(f"Agent: {result['output']}\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Multi-Language Support
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🌍 EXAMPLE 3: Multi-Language Support")
    print("=" * 60 + "\n")

    multilang = MultiLanguageAgent()

    languages = [
        ("English", "Maria", "friendly and helpful"),
        ("Spanish", "Carlos", "formal and respectful"),
        ("French", "Sophie", "warm and welcoming"),
    ]

    for language, name, tone in languages:
        result = await multilang.process(
            input="What's the weather like today?",
            instruction_params={
                "language": language,
                "user_name": name,
                "tone": tone
            }
        )
        print(f"🗣️ Language: {language} | User: {name}")
        print(f"   Response: {result['output'][:100]}...\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 4: Role-Based Agent
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🎭 EXAMPLE 4: Role-Based Agent")
    print("=" * 60 + "\n")

    roleplay = RoleBasedAgent()

    # Sherlock Holmes
    result = await roleplay.process(
        input="There's been a mysterious theft at the museum. What do you make of it?",
        instruction_params={
            "role": "Sherlock Holmes, the famous detective",
            "character_description": """
            - Brilliant deductive reasoning
            - Somewhat arrogant but well-meaning
            - Observes tiny details others miss
            - Often says "Elementary, my dear Watson"
            - Victorian-era speaking style
            """
        }
    )

    print(f"🔍 Character: Sherlock Holmes\n")
    print(f"Question: There's been a theft at the museum.\n")
    print(f"Holmes: {result['output']}\n")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Dynamic Instructions Best Practices
# =============================================================================
#
# 1. USE DESCRIPTIVE PLACEHOLDER NAMES
#    Good: {user_name}, {company_policy}, {current_date}
#    Bad:  {x}, {p1}, {thing}
#
# 2. PLACEHOLDER BEHAVIOR
#    By default, unmatched {placeholders} are left as-is (safe for
#    instructions with example formats or code snippets).
#
#    To enforce that all placeholders must be provided, enable strict mode:
#
#    class Config:
#        strict_instruction_params = True  # Raises InstructionKeyError if missing
#
# 3. KEEP INSTRUCTIONS STRUCTURED
#    Format your docstring clearly with sections:
#
#    """
#    You are a {role}.
#
#    Context:
#    - {context_item_1}
#    - {context_item_2}
#
#    Rules:
#    - Always do X
#    - Never do Y
#    """
#
# 4. DON'T OVERUSE PLACEHOLDERS
#    If you find yourself with 10+ placeholders, consider:
#    - Using different agent classes instead
#    - Putting context in the user message instead of instructions
#
# 5. SANITIZE USER INPUT
#    If placeholder values come from user input, validate them!
#    Malicious users could try to inject instructions.
#
# 6. COMBINE WITH TOOLS
#    Dynamic instructions work great with tools:
#
#    class ContextualAgent(OpenAIAgent):
#        """You are helping {user_name} with their {task}."""
#
#        async def search(self, query: str) -> dict:
#            return {"results": [...]}

