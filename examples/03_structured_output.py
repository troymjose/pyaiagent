"""
03_structured_output.py — Typed Responses with Pydantic
=======================================================

Instead of parsing text responses manually, pyaiagent can return
structured data as Pydantic models. Perfect for building reliable
data pipelines and APIs!

What you'll learn:
  • How to define response schemas with Pydantic
  • How to configure `text_format` for structured output
  • How to access the parsed model via `result["output_parsed"]`

Key concept:
  Set `text_format = YourPydanticModel` in the agent's Config class.
  The agent will return responses matching that schema, and you get
  a fully validated Pydantic instance.

Prerequisites:
  pip install pyaiagent pydantic
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/03_structured_output.py
"""

import asyncio

from pydantic import BaseModel, Field

from pyaiagent import OpenAIAgent


# =============================================================================
# Example 1: Movie Review — Simple Structured Output
# =============================================================================

class MovieReview(BaseModel):
    """A structured movie review."""

    title: str = Field(description="The movie's title")
    year: int = Field(description="Year the movie was released")
    rating: int = Field(ge=1, le=10, description="Rating from 1 to 10")
    summary: str = Field(description="A 2-3 sentence summary")
    pros: list[str] = Field(description="Positive aspects (2-4 items)")
    cons: list[str] = Field(description="Negative aspects (1-3 items)")
    recommended: bool = Field(description="Would you recommend this movie?")


class MovieCritic(OpenAIAgent):
    """
    You are a professional movie critic with years of experience.
    Analyze movies thoughtfully and provide balanced, insightful reviews.
    Be specific about what works and what doesn't.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.7  # Slightly creative for engaging reviews
        text_format = MovieReview  # ← This enables structured output!


# =============================================================================
# Example 2: Sentiment Analysis — Fast Classification
# =============================================================================

class SentimentAnalysis(BaseModel):
    """Sentiment classification result."""

    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    key_phrases: list[str] = Field(description="Key phrases that influenced the sentiment")
    explanation: str = Field(description="Brief explanation of the classification")


class SentimentClassifier(OpenAIAgent):
    """
    You are a sentiment analysis expert.
    Classify the sentiment of the given text accurately.
    Identify key phrases that contribute to the sentiment.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.0  # Maximum consistency for classification
        max_output_tokens = 500  # Keep it fast
        text_format = SentimentAnalysis


# =============================================================================
# Example 3: Product Analysis — Nested Models
# =============================================================================

class ProductFeature(BaseModel):
    """A single product feature."""

    name: str = Field(description="Feature name")
    description: str = Field(description="What this feature does")
    importance: str = Field(description="high, medium, or low")


class ProductAnalysis(BaseModel):
    """Comprehensive product analysis."""

    product_name: str = Field(description="Full product name")
    category: str = Field(description="Product category")
    target_audience: str = Field(description="Who this product is best for")
    price_range: str = Field(description="budget, mid-range, or premium")
    key_features: list[ProductFeature] = Field(description="Top 3-5 features")
    overall_score: float = Field(ge=0.0, le=5.0, description="Score out of 5.0")
    verdict: str = Field(description="One-sentence final verdict")


class ProductAnalyzer(OpenAIAgent):
    """
    You are a tech product analyst.
    Analyze products objectively, considering features, value, and target market.
    Be practical and balanced in your assessments.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.3
        text_format = ProductAnalysis


# =============================================================================
# Example 4: Email Triage — Practical Business Use Case
# =============================================================================

class EmailClassification(BaseModel):
    """Email triage classification."""

    category: str = Field(description="urgent, important, normal, or spam")
    priority: int = Field(ge=1, le=5, description="Priority 1-5 (1=highest)")
    requires_response: bool = Field(description="Does this need a reply?")
    suggested_action: str = Field(description="Recommended next step")
    key_points: list[str] = Field(description="Main points from the email")
    sentiment: str = Field(description="positive, neutral, or negative")


class EmailTriageAgent(OpenAIAgent):
    """
    You are an email triage assistant for a busy executive.
    Analyze emails and classify them to help prioritize the inbox.
    Be accurate and practical in your classifications.
    Focus on actionable insights.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.1  # Consistent classification
        text_format = EmailClassification


# =============================================================================
# Run the Examples
# =============================================================================

async def main() -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Movie Review
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🎬 EXAMPLE 1: Movie Review (Structured Output)")
    print("=" * 60 + "\n")

    critic = MovieCritic()
    result = await critic.process(
        input="Review the movie 'Inception' by Christopher Nolan"
    )

    # Access the parsed Pydantic model — fully typed!
    review: MovieReview = result["output_parsed"]

    print(f"🎬 {review.title} ({review.year})")
    print(f"⭐ Rating: {review.rating}/10")
    print(f"\n📝 Summary:\n   {review.summary}")
    print(f"\n✅ Pros:")
    for pro in review.pros:
        print(f"   • {pro}")
    print(f"\n❌ Cons:")
    for con in review.cons:
        print(f"   • {con}")
    print(f"\n👍 Recommended: {'Yes!' if review.recommended else 'No'}")

    print("\n" + "-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Sentiment Analysis (Batch Processing)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("📊 EXAMPLE 2: Sentiment Analysis (Batch)")
    print("=" * 60 + "\n")

    classifier = SentimentClassifier()

    texts = [
        "I absolutely love this product! Best purchase I've ever made!",
        "The service was okay, nothing special but not bad either.",
        "Terrible experience. The product broke after one day. Never again!",
    ]

    for text in texts:
        result = await classifier.process(input=f"Analyze: {text}")
        analysis: SentimentAnalysis = result["output_parsed"]

        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(
            analysis.sentiment, "🤔"
        )

        print(f'"{text[:50]}..."')
        print(f"   {emoji} Sentiment: {analysis.sentiment.upper()} ({analysis.confidence:.0%})")
        print(f"   📌 Key: {', '.join(analysis.key_phrases[:3])}")
        print()

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Product Analysis (Nested Models)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("📱 EXAMPLE 3: Product Analysis (Nested Models)")
    print("=" * 60 + "\n")

    analyzer = ProductAnalyzer()
    result = await analyzer.process(
        input="Analyze the Apple AirPods Pro (2nd generation)"
    )

    analysis: ProductAnalysis = result["output_parsed"]

    print(f"📱 {analysis.product_name}")
    print(f"📂 Category: {analysis.category}")
    print(f"👥 Target: {analysis.target_audience}")
    print(f"💰 Price: {analysis.price_range}")
    print(f"⭐ Score: {analysis.overall_score}/5.0")
    print(f"\n🔑 Key Features:")
    for feature in analysis.key_features:
        importance_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            feature.importance.lower(), "⚪"
        )
        print(f"   {importance_emoji} {feature.name}: {feature.description}")
    print(f"\n📋 Verdict: {analysis.verdict}")

    print("\n" + "-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 4: Email Triage
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("📧 EXAMPLE 4: Email Triage")
    print("=" * 60 + "\n")

    triage = EmailTriageAgent()

    sample_email = """
    Subject: URGENT: Production System Down

    Hi Team,

    We're experiencing a critical outage affecting our main production system.
    All customer-facing services are currently unavailable.

    I need the on-call team to join the war room immediately.
    This started at 2:45 PM and we've isolated it to a database issue.

    Revenue impact is approximately $50k/hour.

    Please respond ASAP.

    Thanks,
    Sarah (VP Engineering)
    """

    result = await triage.process(
        input=f"Classify this email:\n{sample_email}"
    )

    email: EmailClassification = result["output_parsed"]

    priority_indicator = "🔴" * (6 - email.priority)  # More dots = higher priority
    print(f"📧 Email Classification:")
    print(f"   Category:  {email.category.upper()}")
    print(f"   Priority:  {priority_indicator} ({email.priority}/5)")
    print(f"   Response:  {'Yes, immediately!' if email.requires_response else 'No'}")
    print(f"   Sentiment: {email.sentiment}")
    print(f"\n   💡 Action: {email.suggested_action}")
    print(f"\n   📌 Key Points:")
    for point in email.key_points:
        print(f"      • {point}")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Tips for Structured Output
# =============================================================================
#
# 1. USE FIELD DESCRIPTIONS
#    The AI uses Field(description=...) to understand what you want.
#    Be specific: "A number from 1-10" is better than "A rating".
#
# 2. USE CONSTRAINTS
#    Pydantic validators help ensure data quality:
#      rating: int = Field(ge=1, le=10)  # Guarantees 1-10
#
# 3. KEEP MODELS FOCUSED
#    Smaller, focused models work better than large complex ones.
#    Break complex outputs into multiple agents if needed.
#
# 4. BATCH SIMILAR REQUESTS
#    If classifying many items, consider batching them in one prompt
#    with a list output model.
#
# 5. ACCESS RAW TEXT TOO
#    The raw text is always available in result["output"] if you need it.
