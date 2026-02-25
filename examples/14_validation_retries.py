"""
14_validation_retries.py — Auto-Retry on Structured Output Validation Failure
==============================================================================

OpenAI's Structured Outputs guarantees JSON schema conformance, but custom
Pydantic validators can still fail. pyaiagent can automatically retry by
sending validation errors back to the LLM, giving it a chance to self-correct.

What you'll learn:
  • How to add custom Pydantic validators to structured outputs
  • How `validation_retries` auto-retries with error feedback to the LLM
  • How messages.llm stays clean while messages.ui shows the full retry history
  • How to handle ValidationRetriesExhaustedError
  • How to disable retries (default) for manual retry handling

Key concept:
  Set `validation_retries = N` in Config to auto-retry up to N times.
  Default is 0 (disabled), so developers who handle retries themselves
  are unaffected.

Prerequisites:
  pip install pyaiagent pydantic
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/14_validation_retries.py
"""

import asyncio

from pydantic import BaseModel, Field, field_validator

from pyaiagent import OpenAIAgent, ValidationRetriesExhaustedError


# =============================================================================
# Example 1: Basic Validation Retries
# =============================================================================

class MovieReview(BaseModel):
    """A movie review with strict validation rules."""

    title: str = Field(description="The movie's title")
    rating: int = Field(description="Rating from 1 to 10")
    summary: str = Field(description="A detailed summary of the movie")

    @field_validator("rating")
    @classmethod
    def rating_must_be_in_range(cls, v):
        if not 1 <= v <= 10:
            raise ValueError("Rating must be between 1 and 10")
        return v

    @field_validator("summary")
    @classmethod
    def summary_must_be_detailed(cls, v):
        if len(v.split()) < 10:
            raise ValueError("Summary must be at least 10 words long")
        return v


class ReviewAgent(OpenAIAgent):
    """
    You are a movie critic. Provide detailed, structured reviews.
    Always give a specific rating between 1 and 10.
    Write detailed summaries of at least 2-3 sentences.
    """

    class Config:
        model = "gpt-4o-mini"
        text_format = MovieReview
        validation_retries = 3  # Retry up to 3 times on validation failure


async def example_basic_retries() -> None:
    """Demonstrate basic validation retries."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Validation Retries")
    print("=" * 60 + "\n")

    agent = ReviewAgent()
    result = await agent.process(input="Review the movie Inception")

    review: MovieReview = result["output_parsed"]
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}/10")
    print(f"Summary: {review.summary}")
    print(f"\nSteps taken: {result['steps']}")
    print(f"Tokens used: {result['tokens']['total_tokens']}")


# =============================================================================
# Example 2: Inspecting Messages After Retries
# =============================================================================

async def example_inspect_messages() -> None:
    """
    Show how messages.llm is clean while messages.ui preserves retry history.

    This is the key design:
      - messages["llm"] is for the LLM — clean, no retry noise
      - messages["ui"]  is for your app — full history with step numbers
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Messages After Retries (llm vs ui)")
    print("=" * 60 + "\n")

    agent = ReviewAgent()
    result = await agent.process(input="Review the movie The Matrix")

    # messages.llm — Clean: only user input + final valid response
    llm_messages = result["messages"]["llm"]
    print(f"messages.llm count: {len(llm_messages)}")
    for msg in llm_messages:
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))[:60]
        print(f"  [{role}] {content}...")

    print()

    # messages.ui — Full history: shows all attempts including failures
    ui_messages = result["messages"]["ui"]
    print(f"messages.ui count: {len(ui_messages)}")
    for msg in ui_messages:
        role = msg.get("role", "unknown")
        step = msg.get("step", "?")
        content = str(msg.get("content", ""))[:60]
        print(f"  [step {step}] [{role}] {content}...")

    if len(ui_messages) > len(llm_messages):
        print(f"\n  -> ui has more messages because it preserves failed attempts")
        print(f"  -> llm is clean for passing to the next process() call")


# =============================================================================
# Example 3: Handling Exhausted Retries
# =============================================================================

class ImpossibleReview(BaseModel):
    """A review model with an intentionally strict validator."""

    title: str
    rating: int

    @field_validator("rating")
    @classmethod
    def must_be_exactly_42(cls, v):
        if v != 42:
            raise ValueError("Rating must be exactly 42 (this is intentionally hard)")
        return v


class StrictAgent(OpenAIAgent):
    """You are a movie critic. Rate movies on a scale of 1-10."""

    class Config:
        model = "gpt-4o-mini"
        text_format = ImpossibleReview
        validation_retries = 2  # Only 2 retries to keep costs low


async def example_exhausted_retries() -> None:
    """Demonstrate handling when all retries are exhausted."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Handling Exhausted Retries")
    print("=" * 60 + "\n")

    agent = StrictAgent()

    try:
        await agent.process(input="Review Inception")
        print("Unexpectedly succeeded!")

    except ValidationRetriesExhaustedError as e:
        print(f"Caught ValidationRetriesExhaustedError")
        print(f"  Message: {str(e)[:100]}...")
        print(f"  Last validation errors: {e.validation_errors[:100]}...")
        print(f"\n  This is expected — the validator requires rating=42,")
        print(f"  but the LLM was told to rate 1-10.")


# =============================================================================
# Example 4: Disabled Retries (Default Behavior)
# =============================================================================

class DefaultAgent(OpenAIAgent):
    """You are a movie critic."""

    class Config:
        model = "gpt-4o-mini"
        text_format = MovieReview
        # validation_retries = 0  ← this is the default, no retries


async def example_disabled_retries() -> None:
    """
    Show the default behavior (validation_retries=0).

    When retries are disabled, the developer handles validation themselves.
    This is backward compatible with existing code.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Disabled Retries (Default)")
    print("=" * 60 + "\n")

    agent = DefaultAgent()
    result = await agent.process(input="Review Inception")

    if result["output_parsed"] is not None:
        review: MovieReview = result["output_parsed"]
        print(f"Title: {review.title}")
        print(f"Rating: {review.rating}/10")
    else:
        print("output_parsed is None — validation may have failed.")
        print("With validation_retries=0, you handle this yourself:")
        print("  - Retry manually with a modified prompt")
        print("  - Fall back to raw output text")
        print(f"  - Raw text: {result['output'][:80]}...")


# =============================================================================
# Example 5: Cross-Field Validation
# =============================================================================

class FlightBooking(BaseModel):
    """A structured flight booking response."""

    departure_city: str
    arrival_city: str
    departure_date: str = Field(description="Date in YYYY-MM-DD format")
    return_date: str = Field(description="Date in YYYY-MM-DD format")
    num_passengers: int

    @field_validator("num_passengers")
    @classmethod
    def passengers_must_be_valid(cls, v):
        if not 1 <= v <= 9:
            raise ValueError("Number of passengers must be between 1 and 9")
        return v

    @field_validator("departure_city", "arrival_city")
    @classmethod
    def cities_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("City name cannot be empty")
        return v.strip()


class BookingAgent(OpenAIAgent):
    """
    You are a flight booking assistant. Extract booking details from user requests.
    Use YYYY-MM-DD format for all dates.
    """

    class Config:
        model = "gpt-4o-mini"
        text_format = FlightBooking
        validation_retries = 3


async def example_cross_field_validation() -> None:
    """Demonstrate validation retries with multiple field validators."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Cross-Field Validation")
    print("=" * 60 + "\n")

    agent = BookingAgent()
    result = await agent.process(
        input="I need to fly from New York to London, departing March 15 2025, "
              "returning March 22 2025, for 2 passengers"
    )

    booking: FlightBooking = result["output_parsed"]
    print(f"From: {booking.departure_city}")
    print(f"To: {booking.arrival_city}")
    print(f"Depart: {booking.departure_date}")
    print(f"Return: {booking.return_date}")
    print(f"Passengers: {booking.num_passengers}")


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    await example_basic_retries()
    await example_inspect_messages()
    await example_exhausted_retries()
    await example_disabled_retries()
    await example_cross_field_validation()

    print("\n" + "=" * 60)
    print("All validation retry examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Summary
# =============================================================================
#
# validation_retries = 0  (default)  No retries, backward compatible.
#                                    Handle validation yourself.
#
# validation_retries = 3  (recommended)  Auto-retry up to 3 times.
#                                        LLM gets error feedback to self-correct.
#
# Key message behavior:
#   messages["llm"]  ->  Clean. Retry artifacts removed.
#                        Safe to pass to next process() call.
#   messages["ui"]   ->  Full history. Shows all attempts with step numbers.
#                        Insert into DB for debugging/analytics.
