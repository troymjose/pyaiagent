"""
Comprehensive unit tests for pyaiagent.openai.client.

Covers: module exports, default client management (set/get/clear),
        set_default guard, AsyncOpenAIClient init/properties/__slots__,
        singleton behavior, aclose (owned/injected/idempotent/shield/finally),
        shutdown (no-op/success/error/raise_on_error), logging, and
        multi-step lifecycle integration.
"""
import asyncio
import logging
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import pyaiagent.openai.client as _client_mod
from pyaiagent.openai.client import (
    AsyncOpenAIClient,
    clear_default_openai_client,
    get_default_openai_client,
    set_default_openai_client,
    shutdown,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class FakeAsyncOpenAI:
    """Spy that records close() calls without real HTTP."""

    def __init__(self):
        self.close_count = 0

    async def close(self):
        self.close_count += 1


class FailingCloseClient:
    """Client whose close() always raises."""

    async def close(self):
        raise OSError("transport error")


class SlowCloseClient:
    """Client with a delayed close for cancellation tests."""

    def __init__(self):
        self.close_completed = asyncio.Event()

    async def close(self):
        await asyncio.sleep(0.05)
        self.close_completed.set()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup():
    """Reset all module state between tests."""
    yield
    AsyncOpenAIClient.delete_all_instances()
    _client_mod._default_openai_client = None


# ──────────────────────────────────────────────────────────────────────────────
# 1. Module Exports
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleExports:

    def test_all_contains_expected_names(self):
        expected = {
            "AsyncOpenAIClient",
            "set_default_openai_client",
            "get_default_openai_client",
            "clear_default_openai_client",
            "shutdown",
        }
        assert set(_client_mod.__all__) == expected

    def test_all_length(self):
        assert len(_client_mod.__all__) == 5

    def test_logger_name(self):
        assert _client_mod.logger.name == "pyaiagent.openai.client"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Default Client Management (set / get / clear)
# ──────────────────────────────────────────────────────────────────────────────

class TestDefaultClientManagement:

    def test_get_default_returns_none_initially(self):
        assert get_default_openai_client() is None

    def test_set_and_get_default(self):
        fake = FakeAsyncOpenAI()
        set_default_openai_client(fake)
        assert get_default_openai_client() is fake

    def test_set_default_overwrites_previous(self):
        fake1 = FakeAsyncOpenAI()
        fake2 = FakeAsyncOpenAI()
        set_default_openai_client(fake1)
        set_default_openai_client(fake2)
        assert get_default_openai_client() is fake2

    def test_clear_default(self):
        set_default_openai_client(FakeAsyncOpenAI())
        clear_default_openai_client()
        assert get_default_openai_client() is None

    def test_clear_then_get_returns_none(self):
        clear_default_openai_client()
        assert get_default_openai_client() is None

    def test_clear_idempotent(self):
        clear_default_openai_client()
        clear_default_openai_client()
        assert get_default_openai_client() is None

    def test_set_after_clear(self):
        fake = FakeAsyncOpenAI()
        set_default_openai_client(FakeAsyncOpenAI())
        clear_default_openai_client()
        set_default_openai_client(fake)
        assert get_default_openai_client() is fake


# ──────────────────────────────────────────────────────────────────────────────
# 3. set_default_openai_client Guard
# ──────────────────────────────────────────────────────────────────────────────

class TestSetDefaultGuard:

    @pytest.mark.asyncio
    async def test_refuses_with_active_instance(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        with pytest.raises(RuntimeError, match="active"):
            set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_allowed_when_no_instances(self):
        set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_allowed_after_shutdown(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        await shutdown()
        set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_allowed_with_only_closed_instances(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        await wrapper.aclose()
        AsyncOpenAIClient.delete_all_instances()
        set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_error_message_content(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        with pytest.raises(RuntimeError) as exc_info:
            set_default_openai_client(FakeAsyncOpenAI())
        msg = str(exc_info.value)
        assert "BEFORE" in msg
        assert "shutdown()" in msg
        assert "active" in msg.lower()

    @pytest.mark.asyncio
    async def test_refuses_with_injected_active_instance(self):
        AsyncOpenAIClient(client=FakeAsyncOpenAI())
        with pytest.raises(RuntimeError, match="active"):
            set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_refuses_with_default_active_instance(self):
        _client_mod._default_openai_client = FakeAsyncOpenAI()
        AsyncOpenAIClient()
        with pytest.raises(RuntimeError, match="active"):
            set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_refuses_from_sync_thread_with_active_on_other_loop(self):
        AsyncOpenAIClient(client=FakeAsyncOpenAI())
        error = {}

        def try_set_from_thread():
            try:
                set_default_openai_client(FakeAsyncOpenAI())
            except RuntimeError as exc:
                error["exc"] = exc

        thread = threading.Thread(target=try_set_from_thread)
        thread.start()
        thread.join(timeout=5)
        assert "exc" in error
        assert "active" in str(error["exc"]).lower()


# ──────────────────────────────────────────────────────────────────────────────
# 4. AsyncOpenAIClient.__init__
# ──────────────────────────────────────────────────────────────────────────────

class TestAsyncOpenAIClientInit:

    @pytest.mark.asyncio
    async def test_explicit_client(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        assert wrapper.client is fake
        assert wrapper.owns_client is False

    @pytest.mark.asyncio
    async def test_default_client(self):
        fake = FakeAsyncOpenAI()
        _client_mod._default_openai_client = fake
        wrapper = AsyncOpenAIClient()
        assert wrapper.client is fake
        assert wrapper.owns_client is False

    @pytest.mark.asyncio
    async def test_internal_client_created(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        assert wrapper.client is fake
        assert wrapper.owns_client is True

    @pytest.mark.asyncio
    async def test_explicit_none_uses_default(self):
        fake = FakeAsyncOpenAI()
        _client_mod._default_openai_client = fake
        wrapper = AsyncOpenAIClient(client=None)
        assert wrapper.client is fake
        assert wrapper.owns_client is False

    @pytest.mark.asyncio
    async def test_explicit_none_no_default_creates_internal(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient(client=None)
        assert wrapper.owns_client is True

    @pytest.mark.asyncio
    async def test_closed_false_initially(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert wrapper.is_active is True

    @pytest.mark.asyncio
    async def test_explicit_takes_priority_over_default(self):
        default = FakeAsyncOpenAI()
        explicit = FakeAsyncOpenAI()
        _client_mod._default_openai_client = default
        wrapper = AsyncOpenAIClient(client=explicit)
        assert wrapper.client is explicit
        assert wrapper.client is not default


# ──────────────────────────────────────────────────────────────────────────────
# 5. __slots__
# ──────────────────────────────────────────────────────────────────────────────

class TestSlots:

    def test_slots_defined(self):
        assert "__slots__" in AsyncOpenAIClient.__dict__

    def test_slot_names(self):
        assert set(AsyncOpenAIClient.__slots__) == {"_client", "_closed", "_owns_client"}

    def test_metaclass_is_per_event_loop_singleton(self):
        from pyaiagent.utils import PerEventLoopSingleton
        assert type(AsyncOpenAIClient) is PerEventLoopSingleton

    @pytest.mark.asyncio
    async def test_declared_slots_accessible(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert hasattr(wrapper, "_client")
        assert hasattr(wrapper, "_closed")
        assert hasattr(wrapper, "_owns_client")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Properties (is_active, owns_client, client)
# ──────────────────────────────────────────────────────────────────────────────

class TestProperties:

    @pytest.mark.asyncio
    async def test_is_active_true_when_open(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert wrapper.is_active is True

    @pytest.mark.asyncio
    async def test_is_active_false_after_close(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        await wrapper.aclose()
        assert wrapper.is_active is False

    @pytest.mark.asyncio
    async def test_owns_client_true_internal(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FakeAsyncOpenAI()):
            wrapper = AsyncOpenAIClient()
        assert wrapper.owns_client is True

    @pytest.mark.asyncio
    async def test_owns_client_false_explicit(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert wrapper.owns_client is False

    @pytest.mark.asyncio
    async def test_owns_client_false_default(self):
        _client_mod._default_openai_client = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient()
        assert wrapper.owns_client is False

    @pytest.mark.asyncio
    async def test_client_returns_underlying(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        assert wrapper.client is fake

    @pytest.mark.asyncio
    async def test_client_consistent_across_accesses(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        assert wrapper.client is wrapper.client

    @pytest.mark.asyncio
    async def test_client_raises_when_closed(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        await wrapper.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            _ = wrapper.client

    @pytest.mark.asyncio
    async def test_client_error_message(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        await wrapper.aclose()
        with pytest.raises(RuntimeError) as exc_info:
            _ = wrapper.client
        assert "Create a new instance" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_owns_client_readable_after_close(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        assert wrapper.owns_client is True
        await wrapper.aclose()
        assert wrapper.owns_client is True

    @pytest.mark.asyncio
    async def test_owns_client_false_readable_after_close(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert wrapper.owns_client is False
        await wrapper.aclose()
        assert wrapper.owns_client is False


# ──────────────────────────────────────────────────────────────────────────────
# 7. Singleton Behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestSingletonBehavior:

    @pytest.mark.asyncio
    async def test_same_loop_returns_same_instance(self):
        fake = FakeAsyncOpenAI()
        wrapper1 = AsyncOpenAIClient(client=fake)
        wrapper2 = AsyncOpenAIClient()
        assert wrapper1 is wrapper2

    @pytest.mark.asyncio
    async def test_get_instance_returns_same(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        assert AsyncOpenAIClient.get_instance() is wrapper

    @pytest.mark.asyncio
    async def test_iter_instances_contains_wrapper(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        instances = AsyncOpenAIClient.iter_instances()
        assert len(instances) == 1
        assert instances[0] is wrapper


# ──────────────────────────────────────────────────────────────────────────────
# 8. aclose()
# ──────────────────────────────────────────────────────────────────────────────

class TestAclose:

    @pytest.mark.asyncio
    async def test_owned_close_calls_underlying(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        await wrapper.aclose()
        assert fake.close_count == 1

    @pytest.mark.asyncio
    async def test_owned_close_marks_closed(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        await wrapper.aclose()
        assert wrapper.is_active is False

    @pytest.mark.asyncio
    async def test_injected_does_not_close_underlying(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        await wrapper.aclose()
        assert fake.close_count == 0

    @pytest.mark.asyncio
    async def test_default_does_not_close_underlying(self):
        fake = FakeAsyncOpenAI()
        _client_mod._default_openai_client = fake
        wrapper = AsyncOpenAIClient()
        await wrapper.aclose()
        assert fake.close_count == 0

    @pytest.mark.asyncio
    async def test_idempotent(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        await wrapper.aclose()
        await wrapper.aclose()
        await wrapper.aclose()
        assert fake.close_count == 1

    @pytest.mark.asyncio
    async def test_removes_from_registry(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        assert AsyncOpenAIClient.get_instance() is not None
        await AsyncOpenAIClient.get_instance().aclose()
        assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_finally_deletes_on_close_error(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            wrapper = AsyncOpenAIClient()
        assert AsyncOpenAIClient.get_instance() is wrapper
        with pytest.raises(OSError, match="transport error"):
            await wrapper.aclose()
        assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_is_active_false_after_aclose(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert wrapper.is_active is True
        await wrapper.aclose()
        assert wrapper.is_active is False

    @pytest.mark.asyncio
    async def test_client_raises_after_aclose(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        await wrapper.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            _ = wrapper.client

    @pytest.mark.asyncio
    async def test_new_instance_after_aclose(self):
        fake1 = FakeAsyncOpenAI()
        wrapper1 = AsyncOpenAIClient(client=fake1)
        await wrapper1.aclose()
        fake2 = FakeAsyncOpenAI()
        wrapper2 = AsyncOpenAIClient(client=fake2)
        assert wrapper1 is not wrapper2
        assert wrapper2.client is fake2

    @pytest.mark.asyncio
    async def test_shield_protects_owned_close_from_cancellation(self):
        slow = SlowCloseClient()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=slow):
            wrapper = AsyncOpenAIClient()

        task = asyncio.create_task(wrapper.aclose())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.1)
        assert slow.close_completed.is_set()

    @pytest.mark.asyncio
    async def test_aclose_uses_asyncio_shield(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        with patch("pyaiagent.openai.client.asyncio.shield", wraps=asyncio.shield) as mock_shield:
            await wrapper.aclose()
        mock_shield.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_does_not_shield_injected(self):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        with patch("pyaiagent.openai.client.asyncio.shield") as mock_shield:
            await wrapper.aclose()
        mock_shield.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# 9. shutdown()
# ──────────────────────────────────────────────────────────────────────────────

class TestShutdown:

    @pytest.mark.asyncio
    async def test_no_op_no_running_loop(self):
        with patch(
            "pyaiagent.openai.client.asyncio.get_running_loop",
            side_effect=RuntimeError,
        ):
            await shutdown()

    @pytest.mark.asyncio
    async def test_no_op_no_instance(self):
        await shutdown()

    @pytest.mark.asyncio
    async def test_closes_owned(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        await shutdown()
        assert fake.close_count == 1

    @pytest.mark.asyncio
    async def test_releases_injected_without_close(self):
        fake = FakeAsyncOpenAI()
        AsyncOpenAIClient(client=fake)
        await shutdown()
        assert fake.close_count == 0

    @pytest.mark.asyncio
    async def test_idempotent(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        await shutdown()
        await shutdown()
        await shutdown()
        assert fake.close_count == 1

    @pytest.mark.asyncio
    async def test_swallows_error_by_default(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        await shutdown()

    @pytest.mark.asyncio
    async def test_reraises_with_flag(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        with pytest.raises(OSError, match="transport error"):
            await shutdown(raise_on_error=True)

    @pytest.mark.asyncio
    async def test_removes_instance_from_registry(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        assert AsyncOpenAIClient.get_instance() is not None
        await shutdown()
        assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_allows_set_default_after_shutdown(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        await shutdown()
        set_default_openai_client(FakeAsyncOpenAI())

    @pytest.mark.asyncio
    async def test_new_instance_after_shutdown(self):
        fake1 = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake1):
            wrapper1 = AsyncOpenAIClient()
        await shutdown()
        fake2 = FakeAsyncOpenAI()
        wrapper2 = AsyncOpenAIClient(client=fake2)
        assert wrapper1 is not wrapper2
        assert wrapper2.client is fake2

    @pytest.mark.asyncio
    async def test_shutdown_removes_even_on_error(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        await shutdown()
        assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_raise_on_error_default_is_false(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        await shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# 10. Logging
# ──────────────────────────────────────────────────────────────────────────────

class TestLogging:

    @pytest.mark.asyncio
    async def test_init_explicit_client_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
            AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert "explicit client" in caplog.text

    @pytest.mark.asyncio
    async def test_init_default_client_log(self, caplog):
        _client_mod._default_openai_client = FakeAsyncOpenAI()
        with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
            AsyncOpenAIClient()
        assert "default client" in caplog.text

    @pytest.mark.asyncio
    async def test_init_internal_client_log(self, caplog):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FakeAsyncOpenAI()):
            with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
                AsyncOpenAIClient()
        assert "new internal client" in caplog.text

    @pytest.mark.asyncio
    async def test_aclose_owned_log(self, caplog):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
        with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
            await wrapper.aclose()
        assert "Owned AsyncOpenAI client closed" in caplog.text

    @pytest.mark.asyncio
    async def test_aclose_released_log(self, caplog):
        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
            await wrapper.aclose()
        assert "external client not closed" in caplog.text

    def test_set_default_debug_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
            set_default_openai_client(FakeAsyncOpenAI())
        assert "Default OpenAI client set" in caplog.text

    def test_clear_default_debug_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="pyaiagent.openai.client"):
            clear_default_openai_client()
        assert "Default OpenAI client cleared" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown_info_log_on_success(self, caplog):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        with caplog.at_level(logging.INFO, logger="pyaiagent.openai.client"):
            await shutdown()
        assert "shutdown complete" in caplog.text
        assert "owns_client" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown_warning_log_on_failure(self, caplog):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        with caplog.at_level(logging.WARNING, logger="pyaiagent.openai.client"):
            await shutdown()
        assert "Error during AsyncOpenAIClient shutdown" in caplog.text
        assert "owns_client" in caplog.text

    @pytest.mark.asyncio
    async def test_set_default_refuses_info_log(self, caplog):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FakeAsyncOpenAI()):
            AsyncOpenAIClient()
        with caplog.at_level(logging.INFO, logger="pyaiagent.openai.client"):
            with pytest.raises(RuntimeError):
                set_default_openai_client(FakeAsyncOpenAI())
        assert "Refusing to set default client" in caplog.text
        assert "1 active instance" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown_info_log_includes_owns_client_true(self, caplog):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FakeAsyncOpenAI()):
            AsyncOpenAIClient()
        with caplog.at_level(logging.INFO, logger="pyaiagent.openai.client"):
            await shutdown()
        assert "owns_client=True" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown_info_log_includes_owns_client_false(self, caplog):
        AsyncOpenAIClient(client=FakeAsyncOpenAI())
        with caplog.at_level(logging.INFO, logger="pyaiagent.openai.client"):
            await shutdown()
        assert "owns_client=False" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown_warning_level_is_warning(self, caplog):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        with caplog.at_level(logging.WARNING, logger="pyaiagent.openai.client"):
            await shutdown()
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "shutdown" in r.message
        ]
        assert len(warning_records) == 1

    @pytest.mark.asyncio
    async def test_shutdown_warning_includes_traceback(self, caplog):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        with caplog.at_level(logging.WARNING, logger="pyaiagent.openai.client"):
            await shutdown()
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "shutdown" in r.message
        ]
        assert warning_records[0].exc_info is not None
        assert warning_records[0].exc_info[0] is OSError

    @pytest.mark.asyncio
    async def test_shutdown_info_level_is_info(self, caplog):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            AsyncOpenAIClient()
        with caplog.at_level(logging.INFO, logger="pyaiagent.openai.client"):
            await shutdown()
        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "shutdown complete" in r.message
        ]
        assert len(info_records) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 11. Integration / Multi-step Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

class TestLifecycleIntegration:

    @pytest.mark.asyncio
    async def test_full_lifecycle_owned(self):
        fake = FakeAsyncOpenAI()
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
            wrapper = AsyncOpenAIClient()
            assert wrapper.is_active
            assert wrapper.owns_client
            _ = wrapper.client
            await shutdown()
            assert not wrapper.is_active
            assert fake.close_count == 1
            assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_full_lifecycle_injected(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        assert wrapper.is_active
        assert not wrapper.owns_client
        _ = wrapper.client
        await shutdown()
        assert not wrapper.is_active
        assert fake.close_count == 0
        assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_set_default_create_shutdown_reconfigure(self):
        fake1 = FakeAsyncOpenAI()
        set_default_openai_client(fake1)
        wrapper1 = AsyncOpenAIClient()
        assert wrapper1.client is fake1
        assert not wrapper1.owns_client
        await shutdown()

        fake2 = FakeAsyncOpenAI()
        set_default_openai_client(fake2)
        wrapper2 = AsyncOpenAIClient()
        assert wrapper2.client is fake2

    @pytest.mark.asyncio
    async def test_aclose_then_recreate(self):
        fake1 = FakeAsyncOpenAI()
        wrapper1 = AsyncOpenAIClient(client=fake1)
        await wrapper1.aclose()

        fake2 = FakeAsyncOpenAI()
        wrapper2 = AsyncOpenAIClient(client=fake2)
        assert wrapper1 is not wrapper2
        assert wrapper2.is_active
        assert wrapper2.client is fake2

    @pytest.mark.asyncio
    async def test_multiple_shutdown_cycles(self):
        for i in range(5):
            fake = FakeAsyncOpenAI()
            with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=fake):
                wrapper = AsyncOpenAIClient()
            assert wrapper.is_active
            await shutdown()
            assert not wrapper.is_active
            assert fake.close_count == 1

    @pytest.mark.asyncio
    async def test_clear_default_does_not_affect_active(self):
        fake = FakeAsyncOpenAI()
        _client_mod._default_openai_client = fake
        wrapper = AsyncOpenAIClient()
        assert wrapper.client is fake
        clear_default_openai_client()
        assert wrapper.client is fake
        assert wrapper.is_active

    @pytest.mark.asyncio
    async def test_shutdown_error_still_allows_recreation(self):
        with patch("pyaiagent.openai.client.AsyncOpenAI", return_value=FailingCloseClient()):
            AsyncOpenAIClient()
        await shutdown()

        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        assert wrapper.is_active
        assert wrapper.client is fake

    @pytest.mark.asyncio
    async def test_get_instance_reflects_lifecycle(self):
        assert AsyncOpenAIClient.get_instance() is None

        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        assert AsyncOpenAIClient.get_instance() is wrapper

        await wrapper.aclose()
        assert AsyncOpenAIClient.get_instance() is None

    @pytest.mark.asyncio
    async def test_shutdown_on_manually_closed_instance_is_no_op(self):
        fake = FakeAsyncOpenAI()
        wrapper = AsyncOpenAIClient(client=fake)
        wrapper._closed = True
        await shutdown()
        assert fake.close_count == 0

    @pytest.mark.asyncio
    async def test_set_default_counts_multiple_active_across_loops(self):
        AsyncOpenAIClient(client=FakeAsyncOpenAI())
        error_holder = {}

        def create_on_other_loop():
            async def _inner():
                AsyncOpenAIClient(client=FakeAsyncOpenAI())
                try:
                    set_default_openai_client(FakeAsyncOpenAI())
                except RuntimeError as exc:
                    error_holder["msg"] = str(exc)
            asyncio.run(_inner())

        thread = threading.Thread(target=create_on_other_loop)
        thread.start()
        thread.join(timeout=5)
        assert "msg" in error_holder
        assert "active" in error_holder["msg"].lower()

    @pytest.mark.asyncio
    async def test_iter_instances_reflects_lifecycle(self):
        assert AsyncOpenAIClient.iter_instances() == []

        wrapper = AsyncOpenAIClient(client=FakeAsyncOpenAI())
        instances = AsyncOpenAIClient.iter_instances()
        assert len(instances) == 1
        assert instances[0] is wrapper

        await wrapper.aclose()
        assert AsyncOpenAIClient.iter_instances() == []
