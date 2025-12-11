"""
09_inheritance_composition.py — Building Agent Hierarchies
==========================================================

As your application grows, you'll want to reuse code between agents.
Python's class inheritance makes this natural — create base agents
with shared tools and config, then specialize them.

What you'll learn:
  • Inheriting tools from base agents
  • Overriding and extending configuration
  • Composing agents for complex workflows
  • Patterns for building agent libraries

Key concept:
  Agent classes follow normal Python inheritance.
  Child agents inherit tools and can override Config.

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/09_inheritance_composition.py
"""

import asyncio
from datetime import datetime

from pyaiagent import OpenAIAgent


# =============================================================================
# Example 1: Base Agent with Shared Tools
# =============================================================================

class BaseAssistant(OpenAIAgent):
    """
    You are a helpful assistant with access to utility tools.
    Use the available tools when they would be helpful.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.3

    async def get_current_time(self) -> dict:
        """Get the current date and time."""
        now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p")
        }

    async def calculate(self, expression: str) -> dict:
        """
        Evaluate a mathematical expression.

        Args:
            expression: A math expression like "2 + 2" or "15 * 3".
        """
        try:
            # Safe eval for basic math (in production, use a proper parser)
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return {"error": "Invalid characters in expression"}
            result = eval(expression)  # Only safe because we filtered chars
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}


# =============================================================================
# Example 2: Specialized Agents (Inherit Tools)
# =============================================================================

class CustomerServiceAgent(BaseAssistant):
    """
    You are a customer service representative for TechCorp.
    Be professional, helpful, and efficient.
    Use the available tools when needed (like checking the time or doing calculations).
    Focus on resolving customer issues quickly.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.4  # Override: slightly more conversational

    # Inherits get_current_time and calculate from BaseAssistant!

    async def lookup_order(self, order_id: str) -> dict:
        """
        Look up a customer order by ID.

        Args:
            order_id: The order ID (e.g., "ORD-12345").
        """
        # Simulated order lookup
        return {
            "order_id": order_id,
            "status": "shipped",
            "estimated_delivery": "2025-01-20",
            "tracking": "TRK-987654"
        }

    async def check_inventory(self, product_id: str) -> dict:
        """
        Check if a product is in stock.

        Args:
            product_id: The product SKU.
        """
        return {
            "product_id": product_id,
            "in_stock": True,
            "quantity": 42,
            "warehouse": "West Coast"
        }


class TechnicalSupportAgent(BaseAssistant):
    """
    You are a technical support specialist.
    Provide detailed, accurate technical guidance.
    Walk users through solutions step by step.
    Use tools when they help solve technical problems.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.2  # Override: more precise for technical work
        max_output_tokens = 4096  # Allow detailed explanations

    # Inherits get_current_time and calculate from BaseAssistant!

    async def run_diagnostic(self, system: str) -> dict:
        """
        Run a diagnostic check on a system.

        Args:
            system: System to check: "network", "database", or "server".
        """
        return {
            "system": system,
            "status": "operational",
            "latency_ms": 45,
            "uptime": "99.97%",
            "last_incident": "2024-12-01"
        }

    async def search_knowledge_base(self, query: str) -> dict:
        """
        Search the technical knowledge base for solutions.

        Args:
            query: The search query.
        """
        return {
            "query": query,
            "results": [
                {"title": "Troubleshooting Guide", "relevance": 0.95},
                {"title": "Configuration Manual", "relevance": 0.82},
                {"title": "FAQ: Common Issues", "relevance": 0.78}
            ]
        }


# =============================================================================
# Example 3: Multi-Level Inheritance
# =============================================================================

class SeniorSupportAgent(TechnicalSupportAgent):
    """
    You are a senior technical support specialist with escalation authority.
    Handle complex issues and can escalate to engineering when needed.
    You have all the tools of a regular tech support agent, plus escalation.
    """

    class Config:
        model = "gpt-4o"  # More capable model for senior role
        temperature = 0.2
        max_steps = 15  # More steps for complex troubleshooting

    # Inherits: get_current_time, calculate, run_diagnostic, search_knowledge_base

    async def escalate_to_engineering(
        self,
        issue_summary: str,
        severity: str = "medium"
    ) -> dict:
        """
        Escalate an issue to the engineering team.

        Args:
            issue_summary: Brief description of the problem.
            severity: Priority level: "low", "medium", "high", or "critical".
        """
        return {
            "escalation_id": "ESC-2025-0042",
            "summary": issue_summary,
            "severity": severity,
            "assigned_team": "Core Platform",
            "eta_response": "2 hours" if severity == "critical" else "24 hours"
        }


# =============================================================================
# Example 4: Composition Pattern (Multiple Specialized Agents)
# =============================================================================

class DispatcherAgent(OpenAIAgent):
    """
    You are a smart dispatcher that routes requests to the right department.
    Analyze the user's request and determine the best department:
    - "customer_service" for orders, billing, general inquiries
    - "technical_support" for technical issues, bugs, system problems
    - "sales" for pricing, new purchases, upgrades

    Respond with ONLY the department name, nothing else.
    """

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.0  # Deterministic routing


async def route_request(user_message: str) -> str:
    """Route a request to the appropriate agent using a dispatcher."""
    dispatcher = DispatcherAgent()

    # Determine the right department
    result = await dispatcher.process(
        input=f"Route this request: {user_message}"
    )
    department = result["output"].strip().lower()

    # Select the appropriate agent
    agents = {
        "customer_service": CustomerServiceAgent(),
        "technical_support": TechnicalSupportAgent(),
    }

    agent = agents.get(department, CustomerServiceAgent())

    # Process with the selected agent
    final_result = await agent.process(input=user_message)
    return f"[{department.upper()}] {final_result['output']}"


# =============================================================================
# Run Examples
# =============================================================================

async def main() -> None:
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Customer Service (inherited tools)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("👤 EXAMPLE 1: Customer Service Agent")
    print("=" * 60 + "\n")

    cs_agent = CustomerServiceAgent()

    # Show inherited tools
    print(f"Available tools: {cs_agent.__tool_names__}")
    print("(Note: get_current_time and calculate are inherited!)\n")

    result = await cs_agent.process(
        input="What time is it? Also, can you check on order ORD-12345?"
    )
    print(f"Agent: {result['output']}\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Technical Support (different specialization)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🔧 EXAMPLE 2: Technical Support Agent")
    print("=" * 60 + "\n")

    tech_agent = TechnicalSupportAgent()

    print(f"Available tools: {tech_agent.__tool_names__}\n")

    result = await tech_agent.process(
        input="Our website is slow. Can you run a diagnostic?"
    )
    print(f"Agent: {result['output']}\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Senior Support (multi-level inheritance)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("👨‍💼 EXAMPLE 3: Senior Support (Multi-Level Inheritance)")
    print("=" * 60 + "\n")

    senior_agent = SeniorSupportAgent()

    print(f"Available tools: {senior_agent.__tool_names__}")
    print("(Inherits from TechnicalSupportAgent AND BaseAssistant!)\n")

    result = await senior_agent.process(
        input="This is a critical issue affecting all customers. "
              "Database queries are timing out. Please escalate to engineering."
    )
    print(f"Agent: {result['output']}\n")

    print("-" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Example 4: Routing/Composition
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("🔀 EXAMPLE 4: Request Routing (Composition)")
    print("=" * 60 + "\n")

    requests = [
        "Where is my order #12345?",
        "The login page is showing a 500 error.",
    ]

    for request in requests:
        print(f"User: {request}")
        response = await route_request(request)
        print(f"Response: {response[:100]}...\n")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Inheritance Patterns Summary
# =============================================================================
#
# 1. TOOL INHERITANCE
#    Child classes automatically get all tools from parent classes.
#    They can add new tools but cannot remove inherited ones.
#
# 2. CONFIG INHERITANCE
#    Config classes are NOT inherited automatically.
#    Define a new Config to override settings, or don't define one
#    to use defaults.
#
# 3. INSTRUCTION OVERRIDE
#    Each class has its own docstring (instructions).
#    Child docstrings completely replace parent docstrings.
#
# 4. COMPOSITION VS INHERITANCE
#    - Use inheritance for "is-a" relationships
#      (SeniorSupport IS-A TechnicalSupport)
#    - Use composition for "uses" relationships
#      (Dispatcher USES CustomerService, TechnicalSupport)
#
# 5. RECOMMENDED STRUCTURE
#
#    BaseAgent (shared utilities)
#    ├── CustomerAgent (CS tools)
#    │   └── VIPCustomerAgent (extra perks)
#    ├── TechAgent (tech tools)
#    │   └── SeniorTechAgent (escalation)
#    └── SalesAgent (sales tools)

