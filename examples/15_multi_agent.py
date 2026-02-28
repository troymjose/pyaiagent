"""
Example 15: Multi-Agent Orchestration
=====================================

Demonstrates the three multi-agent patterns:

1. **class Agents:** — LLM-driven delegation via an inner class
2. **pipeline()** — Developer-driven sequential chaining
3. **parallel()** — Developer-driven concurrent fan-out
4. **team()** — Ad-hoc orchestration without defining a class
5. **handoff()** — Customising handoff descriptions

Run:
    export OPENAI_API_KEY="sk-..."
    python examples/15_multi_agent.py
"""
import asyncio
from pyaiagent import OpenAIAgent, handoff, team, pipeline, parallel


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Define specialist agents (single-responsibility)
# ─────────────────────────────────────────────────────────────────────────────

class Researcher(OpenAIAgent):
    """You are a research specialist. Given a topic, provide a concise
    summary of key facts, recent developments, and important context.
    Keep your response under 200 words."""

    class Config:
        model = "gpt-4o-mini"


class Writer(OpenAIAgent):
    """You are a professional writer. Given research notes, write a
    polished, engaging article. Keep it under 300 words."""

    class Config:
        model = "gpt-4o-mini"


class Editor(OpenAIAgent):
    """You are a meticulous editor. Review the given text for clarity,
    grammar, and style. Return the improved version."""

    class Config:
        model = "gpt-4o-mini"


class FactChecker(OpenAIAgent):
    """You are a fact-checking specialist. Verify claims in the given
    text and flag any that seem inaccurate or need citations."""

    class Config:
        model = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: class Agents: — LLM decides when to delegate
# ─────────────────────────────────────────────────────────────────────────────

class ContentManager(OpenAIAgent):
    """You manage content creation. When asked to write about a topic:
    1. First delegate research to the researcher
    2. Then delegate writing to the writer based on the research
    3. Provide the final article to the user"""

    class Config:
        model = "gpt-4o-mini"
        max_steps = 10

    class Agents:
        researcher = Researcher
        writer = Writer


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1b: class Agents: with handoff() for custom descriptions
# ─────────────────────────────────────────────────────────────────────────────

class QualityManager(OpenAIAgent):
    """You ensure content quality. Use the editor for style improvements
    and the fact_checker to verify accuracy."""

    class Config:
        model = "gpt-4o-mini"
        max_steps = 10

    class Agents:
        editor = Editor
        fact_checker = handoff(FactChecker, description="Verify factual accuracy of text")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1c: Inheritance — child agent inherits parent's agents
# ─────────────────────────────────────────────────────────────────────────────

class FullContentTeam(ContentManager):
    """You manage the full content pipeline: research, write, edit,
    and fact-check. Delegate to each specialist as needed."""

    class Agents:
        editor = Editor
        fact_checker = FactChecker


async def demo_class_agents():
    print("=" * 60)
    print("Pattern 1: class Agents: (LLM-driven)")
    print("=" * 60)

    print("\nAgent definition:")
    print(ContentManager.get_definition())

    # In a real app, you would run:
    # result = await ContentManager().process(input="Write about quantum computing")
    # print(result["output"])

    print("\nFull team inherits research + writer, adds editor + fact_checker:")
    defn = FullContentTeam.get_definition()
    print(f"  Agents: {list(defn.get('agents', {}).keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: pipeline() — sequential chaining
# ─────────────────────────────────────────────────────────────────────────────

async def demo_pipeline():
    print("\n" + "=" * 60)
    print("Pattern 2: pipeline() (sequential)")
    print("=" * 60)
    print("\nResearcher → Writer → Editor")
    print("Each agent's output becomes the next agent's input.")
    print()

    # In a real app:
    # result = await pipeline(
    #     [Researcher, Writer, Editor],
    #     input="Quantum computing breakthroughs in 2024",
    # )
    # print(f"Final article: {result['output']}")
    # print(f"Total tokens: {result['tokens']}")

    print("  pipeline([Researcher, Writer, Editor], input='...')")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: parallel() — concurrent fan-out
# ─────────────────────────────────────────────────────────────────────────────

async def demo_parallel():
    print("\n" + "=" * 60)
    print("Pattern 3: parallel() (concurrent)")
    print("=" * 60)
    print("\nEditor + FactChecker run simultaneously on the same text.")
    print()

    # In a real app:
    # results = await parallel(
    #     [Editor, FactChecker],
    #     input="Some article text to review...",
    # )
    # edited = results[0]["output"]
    # fact_check = results[1]["output"]
    # print(f"Edited version: {edited}")
    # print(f"Fact check: {fact_check}")

    print("  parallel([Editor, FactChecker], input='...')")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: team() — ad-hoc orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def demo_team():
    print("\n" + "=" * 60)
    print("Pattern 4: team() (ad-hoc)")
    print("=" * 60)
    print("\nCreate an orchestrator on the fly — no class needed.")
    print()

    # In a real app:
    # result = await team(
    #     "You coordinate research and writing.",
    #     agents=[Researcher, Writer],
    #     input="Write about quantum computing",
    # )
    # print(result["output"])

    print('  team("You coordinate.", agents=[Researcher, Writer], input="...")')


# ─────────────────────────────────────────────────────────────────────────────

async def main():
    await demo_class_agents()
    await demo_pipeline()
    await demo_parallel()
    await demo_team()

    print("\n" + "=" * 60)
    print("All patterns demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
