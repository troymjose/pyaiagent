"""
Comprehensive unit tests for pyaiagent.utils.PerEventLoopSingleton.

Covers: metaclass setup, instance creation, singleton guarantee,
        recursive construction, construction failure recovery,
        delete_instance, get_instance, iter_instances, delete_all_instances,
        multi-loop behavior, and thread safety.
"""
import asyncio
import threading

import pytest

from pyaiagent.utils import PerEventLoopSingleton

# Access the sentinel for testing internal registry states
import pyaiagent.utils as _utils_mod

_IN_PROGRESS = _utils_mod._IN_PROGRESS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class DummySingleton(metaclass=PerEventLoopSingleton):
    def __init__(self, value=None):
        self.value = value


class AnotherSingleton(metaclass=PerEventLoopSingleton):
    def __init__(self):
        pass


class FailingInit(metaclass=PerEventLoopSingleton):
    def __init__(self):
        raise ValueError("init failed")


class RecursiveInit(metaclass=PerEventLoopSingleton):
    def __init__(self):
        RecursiveInit()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup():
    yield
    DummySingleton.delete_all_instances()
    AnotherSingleton.delete_all_instances()
    FailingInit.delete_all_instances()
    RecursiveInit.delete_all_instances()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Metaclass Setup
# ──────────────────────────────────────────────────────────────────────────────

class TestMetaclassSetup:

    def test_class_has_lock(self):
        assert hasattr(DummySingleton, "_lock")
        assert isinstance(DummySingleton._lock, type(threading.Lock()))

    def test_class_has_instances_per_loop(self):
        assert hasattr(DummySingleton, "_instances_per_loop")

    def test_separate_classes_have_independent_registries(self):
        assert DummySingleton._instances_per_loop is not AnotherSingleton._instances_per_loop
        assert DummySingleton._lock is not AnotherSingleton._lock

    def test_metaclass_type(self):
        assert type(DummySingleton) is PerEventLoopSingleton

    def test_class_name_preserved(self):
        assert DummySingleton.__name__ == "DummySingleton"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Instance Creation
# ──────────────────────────────────────────────────────────────────────────────

class TestInstanceCreation:

    def test_raises_without_running_loop(self):
        with pytest.raises(RuntimeError, match="requires a running event loop"):
            DummySingleton()

    def test_error_message_contains_class_name(self):
        with pytest.raises(RuntimeError, match="DummySingleton"):
            DummySingleton()

    def test_no_loop_error_chains_original(self):
        with pytest.raises(RuntimeError) as exc_info:
            DummySingleton()
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_creates_instance_with_running_loop(self):
        inst = DummySingleton()
        assert isinstance(inst, DummySingleton)

    @pytest.mark.asyncio
    async def test_preserves_constructor_args(self):
        inst = DummySingleton(value=42)
        assert inst.value == 42

    @pytest.mark.asyncio
    async def test_instance_is_correct_type(self):
        inst = DummySingleton()
        assert type(inst) is DummySingleton


# ──────────────────────────────────────────────────────────────────────────────
# 3. Singleton Guarantee
# ──────────────────────────────────────────────────────────────────────────────

class TestSingletonGuarantee:

    @pytest.mark.asyncio
    async def test_same_loop_returns_same_instance(self):
        inst1 = DummySingleton()
        inst2 = DummySingleton()
        assert inst1 is inst2

    @pytest.mark.asyncio
    async def test_second_call_ignores_args(self):
        inst1 = DummySingleton(value=1)
        inst2 = DummySingleton(value=2)
        assert inst2 is inst1
        assert inst2.value == 1

    @pytest.mark.asyncio
    async def test_different_classes_same_loop_are_independent(self):
        d = DummySingleton()
        a = AnotherSingleton()
        assert d is not a

    @pytest.mark.asyncio
    async def test_many_calls_all_return_same(self):
        instances = [DummySingleton() for _ in range(100)]
        assert all(inst is instances[0] for inst in instances)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Recursive Construction
# ──────────────────────────────────────────────────────────────────────────────

class TestRecursiveConstruction:

    @pytest.mark.asyncio
    async def test_recursive_construction_raises(self):
        with pytest.raises(RuntimeError, match="Recursive construction"):
            RecursiveInit()

    @pytest.mark.asyncio
    async def test_recursive_error_contains_class_name(self):
        with pytest.raises(RuntimeError, match="RecursiveInit"):
            RecursiveInit()

    @pytest.mark.asyncio
    async def test_recursive_failure_cleans_registry(self):
        with pytest.raises(RuntimeError):
            RecursiveInit()
        assert RecursiveInit.get_instance() is None
        assert RecursiveInit.iter_instances() == []
        loop = asyncio.get_running_loop()
        assert loop not in RecursiveInit._instances_per_loop


# ──────────────────────────────────────────────────────────────────────────────
# 5. Construction Failure
# ──────────────────────────────────────────────────────────────────────────────

class TestConstructionFailure:

    @pytest.mark.asyncio
    async def test_failure_propagates_exception(self):
        with pytest.raises(ValueError, match="init failed"):
            FailingInit()

    @pytest.mark.asyncio
    async def test_failure_clears_in_progress_marker(self):
        with pytest.raises(ValueError):
            FailingInit()
        loop = asyncio.get_running_loop()
        assert loop not in FailingInit._instances_per_loop

    @pytest.mark.asyncio
    async def test_registry_clean_after_failure(self):
        with pytest.raises(ValueError):
            FailingInit()
        assert FailingInit.get_instance() is None
        assert FailingInit.iter_instances() == []

    @pytest.mark.asyncio
    async def test_retry_after_failure_still_fails(self):
        with pytest.raises(ValueError):
            FailingInit()
        with pytest.raises(ValueError):
            FailingInit()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Multi-Loop Behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiLoop:

    @pytest.mark.asyncio
    async def test_different_loops_produce_different_instances(self):
        inst1 = DummySingleton(value="loop1")
        other_inst = {}

        def create_on_new_loop():
            async def _inner():
                other_inst["inst"] = DummySingleton(value="loop2")
            asyncio.run(_inner())

        thread = threading.Thread(target=create_on_new_loop)
        thread.start()
        thread.join()

        assert "inst" in other_inst
        assert inst1 is not other_inst["inst"]
        assert inst1.value == "loop1"
        assert other_inst["inst"].value == "loop2"

    @pytest.mark.asyncio
    async def test_concurrent_creation_from_threads(self):
        results = {}
        barrier = threading.Barrier(2)

        def create_in_thread(key):
            async def _inner():
                barrier.wait(timeout=2)
                results[key] = id(DummySingleton())
            asyncio.run(_inner())

        t1 = threading.Thread(target=create_in_thread, args=("t1",))
        t2 = threading.Thread(target=create_in_thread, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["t1"] != results["t2"]


# ──────────────────────────────────────────────────────────────────────────────
# 7. delete_instance
# ──────────────────────────────────────────────────────────────────────────────

class TestDeleteInstance:

    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self):
        DummySingleton()
        assert DummySingleton.delete_instance() is True

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self):
        assert DummySingleton.delete_instance() is False

    def test_delete_no_running_loop_returns_false(self):
        assert DummySingleton.delete_instance() is False

    @pytest.mark.asyncio
    async def test_delete_in_progress_returns_false(self):
        loop = asyncio.get_running_loop()
        with DummySingleton._lock:
            DummySingleton._instances_per_loop[loop] = _IN_PROGRESS
        try:
            assert DummySingleton.delete_instance() is False
        finally:
            with DummySingleton._lock:
                DummySingleton._instances_per_loop.pop(loop, None)

    @pytest.mark.asyncio
    async def test_delete_makes_instance_recreatable(self):
        inst1 = DummySingleton(value="first")
        DummySingleton.delete_instance()
        inst2 = DummySingleton(value="second")
        assert inst1 is not inst2
        assert inst2.value == "second"

    @pytest.mark.asyncio
    async def test_delete_idempotent(self):
        DummySingleton()
        assert DummySingleton.delete_instance() is True
        assert DummySingleton.delete_instance() is False


# ──────────────────────────────────────────────────────────────────────────────
# 8. get_instance
# ──────────────────────────────────────────────────────────────────────────────

class TestGetInstance:

    @pytest.mark.asyncio
    async def test_get_existing_instance(self):
        inst = DummySingleton()
        assert DummySingleton.get_instance() is inst

    @pytest.mark.asyncio
    async def test_get_with_explicit_loop(self):
        inst = DummySingleton()
        loop = asyncio.get_running_loop()
        assert DummySingleton.get_instance(loop) is inst

    @pytest.mark.asyncio
    async def test_get_returns_none_when_empty(self):
        assert DummySingleton.get_instance() is None

    def test_get_returns_none_no_running_loop(self):
        assert DummySingleton.get_instance() is None

    @pytest.mark.asyncio
    async def test_get_with_explicit_loop_not_found(self):
        other_loop = asyncio.new_event_loop()
        try:
            assert DummySingleton.get_instance(other_loop) is None
        finally:
            other_loop.close()

    @pytest.mark.asyncio
    async def test_get_returns_none_for_in_progress(self):
        loop = asyncio.get_running_loop()
        with DummySingleton._lock:
            DummySingleton._instances_per_loop[loop] = _IN_PROGRESS
        try:
            assert DummySingleton.get_instance() is None
            assert DummySingleton.get_instance(loop) is None
        finally:
            with DummySingleton._lock:
                DummySingleton._instances_per_loop.pop(loop, None)

    @pytest.mark.asyncio
    async def test_get_does_not_create_instance(self):
        assert DummySingleton.get_instance() is None
        assert DummySingleton.iter_instances() == []

    @pytest.mark.asyncio
    async def test_get_returns_none_for_different_class(self):
        DummySingleton()
        assert AnotherSingleton.get_instance() is None


# ──────────────────────────────────────────────────────────────────────────────
# 9. iter_instances
# ──────────────────────────────────────────────────────────────────────────────

class TestIterInstances:

    @pytest.mark.asyncio
    async def test_empty_registry(self):
        assert DummySingleton.iter_instances() == []

    @pytest.mark.asyncio
    async def test_returns_existing_instance(self):
        inst = DummySingleton()
        result = DummySingleton.iter_instances()
        assert len(result) == 1
        assert result[0] is inst

    @pytest.mark.asyncio
    async def test_excludes_in_progress(self):
        loop = asyncio.get_running_loop()
        with DummySingleton._lock:
            DummySingleton._instances_per_loop[loop] = _IN_PROGRESS
        try:
            assert DummySingleton.iter_instances() == []
        finally:
            with DummySingleton._lock:
                DummySingleton._instances_per_loop.pop(loop, None)

    @pytest.mark.asyncio
    async def test_returns_list_snapshot(self):
        DummySingleton()
        result = DummySingleton.iter_instances()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_snapshot_is_safe_to_modify(self):
        DummySingleton()
        result = DummySingleton.iter_instances()
        result.clear()
        assert len(DummySingleton.iter_instances()) == 1

    @pytest.mark.asyncio
    async def test_separate_classes_independent(self):
        DummySingleton()
        AnotherSingleton()
        assert len(DummySingleton.iter_instances()) == 1
        assert len(AnotherSingleton.iter_instances()) == 1

    def test_iter_no_running_loop_returns_empty(self):
        assert DummySingleton.iter_instances() == []

    @pytest.mark.asyncio
    async def test_iter_mixed_in_progress_and_real(self):
        inst = DummySingleton()
        other_loop = asyncio.new_event_loop()
        try:
            with DummySingleton._lock:
                DummySingleton._instances_per_loop[other_loop] = _IN_PROGRESS
            result = DummySingleton.iter_instances()
            assert len(result) == 1
            assert result[0] is inst
        finally:
            with DummySingleton._lock:
                DummySingleton._instances_per_loop.pop(other_loop, None)
            other_loop.close()


# ──────────────────────────────────────────────────────────────────────────────
# 10. delete_all_instances
# ──────────────────────────────────────────────────────────────────────────────

class TestDeleteAllInstances:

    @pytest.mark.asyncio
    async def test_clears_all(self):
        DummySingleton()
        assert len(DummySingleton.iter_instances()) == 1
        DummySingleton.delete_all_instances()
        assert DummySingleton.iter_instances() == []

    @pytest.mark.asyncio
    async def test_no_op_when_empty(self):
        DummySingleton.delete_all_instances()
        assert DummySingleton.iter_instances() == []

    @pytest.mark.asyncio
    async def test_instance_recreatable_after_clear(self):
        inst1 = DummySingleton(value="before")
        DummySingleton.delete_all_instances()
        inst2 = DummySingleton(value="after")
        assert inst1 is not inst2
        assert inst2.value == "after"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Module Exports
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleExports:

    def test_all_contains_metaclass(self):
        from pyaiagent import utils
        assert "PerEventLoopSingleton" in utils.__all__

    def test_all_length(self):
        from pyaiagent import utils
        assert len(utils.__all__) == 1
