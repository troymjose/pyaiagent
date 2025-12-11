"""
02_tools_basic.py — Agents with Tools (Superpowers!)
====================================================

Tools let your agent DO things — call APIs, query databases, run calculations,
or interact with external systems. This is where agents become truly powerful.

What you'll learn:
  • How to define tools as async methods
  • How docstrings become tool descriptions
  • How type hints become parameter schemas
  • How the AI decides when to use tools

Key concepts:
  • Tools are just async methods on your agent class
  • Method name → tool name
  • Method docstring → tool description (AI reads this!)
  • Type hints → JSON Schema for parameters

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/02_tools_basic.py
"""

import asyncio
from datetime import datetime
from typing import Literal

from pyaiagent import OpenAIAgent


# =============================================================================
# Example 1: Single Tool — Weather Agent
# =============================================================================

class WeatherAgent(OpenAIAgent):
    """
    You are a helpful weather assistant.
    Use the get_weather tool to fetch current weather for any city the user asks about.
    Always provide helpful context about the weather conditions.
    If the user asks about multiple cities, check each one.
    """

    async def get_weather(self, city: str) -> dict:
        """
        Get the current weather for a city.

        Args:
            city: The name of the city (e.g., "New York", "Tokyo", "London").
        """
        # In a real app, you'd call a weather API like OpenWeatherMap.
        # For this demo, we return mock data.

        mock_data = {
            "new york": {"temp": "18°C", "condition": "Partly Cloudy", "humidity": "65%"},
            "tokyo": {"temp": "22°C", "condition": "Sunny", "humidity": "55%"},
            "london": {"temp": "12°C", "condition": "Rainy", "humidity": "80%"},
            "paris": {"temp": "15°C", "condition": "Overcast", "humidity": "70%"},
            "sydney": {"temp": "26°C", "condition": "Clear", "humidity": "45%"},
        }

        weather = mock_data.get(city.lower(), {
            "temp": "20°C",
            "condition": "Clear",
            "humidity": "60%"
        })

        return {
            "city": city,
            "temperature": weather["temp"],
            "condition": weather["condition"],
            "humidity": weather["humidity"],
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }


# =============================================================================
# Example 2: Multiple Tools — Travel Assistant
# =============================================================================

class TravelAssistant(OpenAIAgent):
    """
    You are a knowledgeable travel planning assistant.
    You can check weather, search for flights, and find hotels.
    Use the appropriate tools to help users plan their trips.
    Be helpful and provide relevant recommendations.
    """

    async def get_weather(self, city: str) -> dict:
        """Get current weather for a travel destination."""
        return {
            "city": city,
            "temperature": "24°C",
            "condition": "Sunny",
            "forecast": "Great weather for travel!"
        }

    async def search_flights(
        self,
        origin: str,
        destination: str,
        date: str,
        passengers: int = 1
    ) -> dict:
        """
        Search for available flights between two cities.

        Args:
            origin: Departure city (e.g., "San Francisco").
            destination: Arrival city (e.g., "Tokyo").
            date: Travel date in YYYY-MM-DD format.
            passengers: Number of passengers (default: 1).
        """
        # In reality, you'd call a flight search API
        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "passengers": passengers,
            "flights": [
                {"airline": "Pacific Air", "price": "$850", "duration": "11h 30m", "departure": "08:00"},
                {"airline": "Ocean Airlines", "price": "$920", "duration": "10h 45m", "departure": "14:30"},
                {"airline": "Star Express", "price": "$780", "duration": "12h 15m", "departure": "22:00"},
            ]
        }

    async def find_hotels(
        self,
        city: str,
        check_in: str,
        check_out: str,
        guests: int = 2
    ) -> dict:
        """
        Find available hotels in a city.

        Args:
            city: The city to search for hotels.
            check_in: Check-in date (YYYY-MM-DD).
            check_out: Check-out date (YYYY-MM-DD).
            guests: Number of guests (default: 2).
        """
        return {
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "hotels": [
                {"name": "Grand Palace Hotel", "price": "$180/night", "rating": "4.8★"},
                {"name": "City Center Inn", "price": "$95/night", "rating": "4.2★"},
                {"name": "Boutique Suites", "price": "$250/night", "rating": "4.9★"},
            ]
        }


# =============================================================================
# Example 3: Type Hints Reference — Calculator Agent
# =============================================================================
#
# This example shows various parameter types and how they map to JSON Schema.
#
# ┌─────────────────────────┬────────────────────────────────────────────────┐
# │ Python Type             │ JSON Schema                                    │
# ├─────────────────────────┼────────────────────────────────────────────────┤
# │ str                     │ {"type": "string"}                             │
# │ int                     │ {"type": "integer"}                            │
# │ float                   │ {"type": "number"}                             │
# │ bool                    │ {"type": "boolean"}                            │
# │ list[str]               │ {"type": "array", "items": {"type": "string"}} │
# │ Literal["a", "b"]       │ {"enum": ["a", "b"]}                           │
# │ Optional[str]           │ Can be string or null                          │
# │ str = "default"         │ Optional parameter with default value          │
# └─────────────────────────┴────────────────────────────────────────────────┘

class CalculatorAgent(OpenAIAgent):
    """
    You are a helpful calculator assistant.
    Use the math tools to solve problems.
    Show your work and explain the calculations.
    """

    async def add(self, a: float, b: float) -> dict:
        """Add two numbers together."""
        return {"operation": "add", "a": a, "b": b, "result": a + b}

    async def multiply(self, a: float, b: float) -> dict:
        """Multiply two numbers."""
        return {"operation": "multiply", "a": a, "b": b, "result": a * b}

    async def calculate_percentage(self, value: float, percentage: float) -> dict:
        """
        Calculate a percentage of a value.

        Args:
            value: The base value.
            percentage: The percentage to calculate (e.g., 15 for 15%).
        """
        result = value * (percentage / 100)
        return {
            "operation": "percentage",
            "value": value,
            "percentage": f"{percentage}%",
            "result": round(result, 2)
        }

    async def convert_currency(
        self,
        amount: float,
        from_currency: Literal["USD", "EUR", "GBP", "JPY"],
        to_currency: Literal["USD", "EUR", "GBP", "JPY"]
    ) -> dict:
        """
        Convert between currencies (mock rates).

        Args:
            amount: The amount to convert.
            from_currency: Source currency code.
            to_currency: Target currency code.
        """
        # Mock exchange rates (relative to USD)
        rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.50}

        # Convert to USD, then to target
        usd_amount = amount / rates[from_currency]
        result = usd_amount * rates[to_currency]

        return {
            "original": f"{amount} {from_currency}",
            "converted": f"{result:.2f} {to_currency}",
            "rate": f"1 {from_currency} = {rates[to_currency]/rates[from_currency]:.4f} {to_currency}"
        }


# =============================================================================
# Run the Examples
# =============================================================================

async def main() -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Weather Agent (single tool)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🌤️  EXAMPLE 1: Weather Agent (Single Tool)")
    print("=" * 60 + "\n")

    weather_agent = WeatherAgent()
    result = await weather_agent.process(
        input="What's the weather like in Tokyo and London right now?"
    )

    print(f"Question: What's the weather in Tokyo and London?\n")
    print(f"Agent: {result['output']}\n")
    print(f"📊 Steps taken: {result['steps']} (tool calls + final response)")

    print("\n" + "-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Travel Assistant (multiple tools)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("✈️  EXAMPLE 2: Travel Assistant (Multiple Tools)")
    print("=" * 60 + "\n")

    travel_agent = TravelAssistant()
    result = await travel_agent.process(
        input="I want to fly from San Francisco to Tokyo on 2025-03-15. "
              "Can you find flights and check the weather there?"
    )

    print(f"Agent: {result['output']}\n")
    print(f"📊 Steps taken: {result['steps']}")

    print("\n" + "-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Calculator Agent (type hints demo)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🧮 EXAMPLE 3: Calculator Agent")
    print("=" * 60 + "\n")

    calc_agent = CalculatorAgent()
    result = await calc_agent.process(
        input="If I have a restaurant bill of $85.50, what's an 18% tip? "
              "And what's the total bill including the tip?"
    )

    print(f"Question: Calculate 18% tip on $85.50 and the total.\n")
    print(f"Agent: {result['output']}\n")

    # Currency conversion
    print("-" * 40 + "\n")
    result2 = await calc_agent.process(
        input="Convert 100 USD to EUR and JPY."
    )
    print(f"Question: Convert $100 to EUR and JPY.\n")
    print(f"Agent: {result2['output']}")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Tool Design Best Practices
# =============================================================================
#
# 1. WRITE CLEAR DOCSTRINGS
#    The AI reads your docstring to understand what the tool does.
#    Be specific about parameters and expected values.
#
# 2. USE TYPE HINTS
#    Type hints are converted to JSON Schema. Without them, the AI
#    won't know what type of data to pass.
#
# 3. RETURN DICTS
#    Always return a dict with structured data. This helps the AI
#    understand and summarize the results.
#
# 4. HANDLE ERRORS GRACEFULLY
#    Return error information in the dict rather than raising exceptions:
#
#      async def risky_operation(self, id: str) -> dict:
#          try:
#              data = await fetch_data(id)
#              return {"success": True, "data": data}
#          except Exception as e:
#              return {"success": False, "error": str(e)}
#
# 5. KEEP TOOLS FOCUSED
#    Each tool should do one thing well. Better to have multiple
#    simple tools than one complex tool.
