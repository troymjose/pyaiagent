"""
12_dependency_injection.py — Injecting Dependencies via __init__
================================================================

Sometimes your agent needs external services — database connections,
API clients, or configuration objects. This example shows how to
inject dependencies via __init__ and when to use this vs instruction_params.

What you'll learn:
  • How to override __init__ for dependency injection
  • When to use __init__ vs instruction_params
  • Patterns for database-connected agents
  • Testing agents with mock dependencies

Key concept:
  Use __init__ for static, per-instance dependencies (DB, API clients).
  Use instruction_params for dynamic, per-request context (user name, date).

Prerequisites:
  pip install pyaiagent
  export OPENAI_API_KEY="sk-..."

Run this example:
  python examples/12_dependency_injection.py
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from pyaiagent import OpenAIAgent


# =============================================================================
# Mock Services (In real code, these would be actual clients)
# =============================================================================

@dataclass
class User:
    """A user record."""
    id: str
    name: str
    email: str
    subscription: str


class MockDatabaseClient:
    """Simulates a database client."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._users = {
            "user-1": User("user-1", "Alice", "alice@example.com", "premium"),
            "user-2": User("user-2", "Bob", "bob@example.com", "free"),
            "user-3": User("user-3", "Charlie", "charlie@example.com", "enterprise"),
        }
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Fetch a user by ID."""
        await asyncio.sleep(0.1)  # Simulate DB latency
        return self._users.get(user_id)
    
    async def get_all_users(self) -> list[User]:
        """Fetch all users."""
        await asyncio.sleep(0.1)
        return list(self._users.values())


class MockCacheClient:
    """Simulates a cache client (Redis-like)."""
    
    def __init__(self):
        self._cache = {}
    
    async def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        self._cache[key] = value


class MockEmailService:
    """Simulates an email service."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sent_emails = []
    
    async def send(self, to: str, subject: str, body: str) -> dict:
        """Send an email."""
        email = {"to": to, "subject": subject, "body": body}
        self.sent_emails.append(email)
        return {"status": "sent", "id": f"email-{len(self.sent_emails)}"}


# =============================================================================
# Example 1: Agent with Database Dependency
# =============================================================================

class UserLookupAgent(OpenAIAgent):
    """
    You are a user lookup assistant.
    Use the available tools to find and display user information.
    Always be helpful and protect user privacy.
    """

    def __init__(self, db: MockDatabaseClient):
        """
        Initialize the agent with a database client.
        
        Args:
            db: Database client for user lookups
        """
        super().__init__()  # Always call super().__init__()!
        self.db = db

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.3

    async def lookup_user(self, user_id: str) -> dict:
        """
        Look up a user by their ID.
        
        Args:
            user_id: The user's unique identifier (e.g., "user-1").
        """
        user = await self.db.get_user(user_id)
        if user:
            return {
                "found": True,
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "subscription": user.subscription
            }
        return {"found": False, "error": f"User {user_id} not found"}

    async def list_all_users(self) -> dict:
        """List all users in the system."""
        users = await self.db.get_all_users()
        return {
            "count": len(users),
            "users": [{"id": u.id, "name": u.name, "subscription": u.subscription} for u in users]
        }


# =============================================================================
# Example 2: Agent with Multiple Dependencies
# =============================================================================

class CustomerServiceAgent(OpenAIAgent):
    """
    You are a customer service agent.
    You can look up customers, check their status, and send notifications.
    Always be professional and helpful.
    """

    def __init__(
        self, 
        db: MockDatabaseClient, 
        cache: MockCacheClient, 
        email_service: MockEmailService
    ):
        """
        Initialize with multiple service dependencies.
        
        Args:
            db: Database client for customer data
            cache: Cache client for session data
            email_service: Email service for notifications
        """
        super().__init__()
        self.db = db
        self.cache = cache
        self.email_service = email_service

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.4

    async def get_customer(self, customer_id: str) -> dict:
        """
        Get customer information.
        
        Args:
            customer_id: The customer's ID.
        """
        # Check cache first
        cached = await self.cache.get(f"customer:{customer_id}")
        if cached:
            return {"source": "cache", "data": cached}
        
        # Fall back to database
        user = await self.db.get_user(customer_id)
        if user:
            return {
                "source": "database",
                "id": user.id,
                "name": user.name,
                "subscription": user.subscription
            }
        return {"error": "Customer not found"}

    async def send_notification(self, customer_id: str, message: str) -> dict:
        """
        Send an email notification to a customer.
        
        Args:
            customer_id: The customer's ID.
            message: The notification message to send.
        """
        user = await self.db.get_user(customer_id)
        if not user:
            return {"error": "Customer not found"}
        
        result = await self.email_service.send(
            to=user.email,
            subject="Notification from Support",
            body=message
        )
        return {"sent_to": user.email, "status": result["status"]}


# =============================================================================
# Example 3: Combining __init__ with instruction_params
# =============================================================================
#
# Best pattern: Use __init__ for STATIC dependencies (services, config)
# and instruction_params for DYNAMIC context (user info, current date).

class PersonalizedSupportAgent(OpenAIAgent):
    """
    You are a support agent helping {customer_name}.
    Their subscription level is {subscription_level}.
    Provide assistance appropriate to their subscription tier.
    Today's date is {current_date}.
    """

    def __init__(self, db: MockDatabaseClient, email_service: MockEmailService):
        """Static dependencies — set once at instantiation."""
        super().__init__()
        self.db = db
        self.email_service = email_service

    class Config:
        model = "gpt-4o-mini"
        temperature = 0.5

    async def check_account_status(self, customer_id: str) -> dict:
        """Check a customer's account status."""
        user = await self.db.get_user(customer_id)
        if user:
            return {"name": user.name, "subscription": user.subscription}
        return {"error": "Not found"}


# =============================================================================
# Example 4: Testing with Mock Dependencies
# =============================================================================

class MockDatabaseForTesting(MockDatabaseClient):
    """A mock database that can be configured for testing."""
    
    def __init__(self):
        super().__init__("mock://test")
        # Override with test data
        self._users = {
            "test-user": User("test-user", "Test User", "test@test.com", "test-plan")
        }


# =============================================================================
# Run Examples
# =============================================================================

async def main():
    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Agent with Database
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("📊 EXAMPLE 1: Agent with Database Dependency")
    print("=" * 60)
    
    # Create the database client (would be real connection in production)
    db = MockDatabaseClient("postgresql://localhost/myapp")
    
    # Inject dependency via __init__
    agent = UserLookupAgent(db=db)
    
    result = await agent.process(input="Find the user with ID user-1")
    print(f"\nQuery: Find user-1")
    print(f"Agent: {result['output']}")
    
    result = await agent.process(input="Show me all users")
    print(f"\nQuery: Show all users")
    print(f"Agent: {result['output']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Agent with Multiple Dependencies
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔧 EXAMPLE 2: Agent with Multiple Dependencies")
    print("=" * 60)
    
    # Create all services
    db = MockDatabaseClient("postgresql://localhost/myapp")
    cache = MockCacheClient()
    email = MockEmailService(api_key="sk-email-key")
    
    # Inject all dependencies
    support_agent = CustomerServiceAgent(
        db=db,
        cache=cache,
        email_service=email
    )
    
    result = await support_agent.process(
        input="Look up customer user-2 and send them a welcome message"
    )
    print(f"\nQuery: Look up user-2 and send welcome")
    print(f"Agent: {result['output']}")
    print(f"Emails sent: {len(email.sent_emails)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Combining __init__ with instruction_params
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 EXAMPLE 3: __init__ + instruction_params")
    print("=" * 60)
    
    # Create agent once with static dependencies
    personalized_agent = PersonalizedSupportAgent(
        db=MockDatabaseClient("postgresql://localhost/myapp"),
        email_service=MockEmailService(api_key="sk-email-key")
    )
    
    # Use instruction_params for per-request context
    from datetime import datetime
    
    result = await personalized_agent.process(
        input="What can you help me with?",
        instruction_params={
            "customer_name": "Alice",
            "subscription_level": "Premium",
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    print(f"\nContext: Alice (Premium)")
    print(f"Agent: {result['output']}")
    
    # Same agent, different customer context
    result = await personalized_agent.process(
        input="What features do I have access to?",
        instruction_params={
            "customer_name": "Bob",
            "subscription_level": "Free",
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    print(f"\nContext: Bob (Free)")
    print(f"Agent: {result['output']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Example 4: Testing with Mocks
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🧪 EXAMPLE 4: Testing with Mock Dependencies")
    print("=" * 60)
    
    # In tests, inject mock dependencies
    test_db = MockDatabaseForTesting()
    test_agent = UserLookupAgent(db=test_db)
    
    result = await test_agent.process(input="Find test-user")
    print(f"\nTest Query: Find test-user")
    print(f"Agent: {result['output']}")
    print("(Using mock database with test data)")
    
    print("\n" + "=" * 60)
    print("✅ All Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# When to Use What — Decision Guide
# =============================================================================
#
# ┌─────────────────────────────────────┬──────────────────────────────────────┐
# │ Use __init__ for:                   │ Use instruction_params for:          │
# ├─────────────────────────────────────┼──────────────────────────────────────┤
# │ • Database connections              │ • Current user's name                │
# │ • API clients (Stripe, Twilio)      │ • Today's date                       │
# │ • Cache clients (Redis)             │ • User preferences                   │
# │ • Configuration objects             │ • Session-specific context           │
# │ • Service instances                 │ • Request-specific data              │
# │                                     │                                      │
# │ Static, expensive to create         │ Dynamic, changes per request         │
# │ Shared across all requests          │ Unique to each request               │
# │ Set once at startup                 │ Set at call time                     │
# └─────────────────────────────────────┴──────────────────────────────────────┘
#
# BEST PRACTICE: Combine both!
#
#   # At startup (once)
#   agent = MyAgent(db=db_client, cache=redis_client)
#
#   # Per request (many times)
#   result = await agent.process(
#       input=user_message,
#       instruction_params={"user_name": request.user.name}
#   )
#
# This gives you:
# ✓ Efficient resource usage (one agent instance)
# ✓ Dependency injection for testability
# ✓ Per-request customization
# ✓ Clean separation of concerns
