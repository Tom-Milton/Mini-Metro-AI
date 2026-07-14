import signal

import pytest
from unittest.mock import Mock

from src.infrastructure import shutdown_manager


class TestShutdownManager:
    @pytest.fixture
    def register_signal(self) -> Mock:
        return Mock()
    
    @pytest.fixture
    def manager(self, register_signal: Mock) -> shutdown_manager.ShutdownManager:
        return shutdown_manager.ShutdownManager(
            register_signal=register_signal,
        )

    def test_start_registers_signal_handlers(self, manager: shutdown_manager.ShutdownManager, register_signal: Mock) -> None:
        manager.start()

        register_signal.assert_any_call(
            signal.SIGINT,
            manager.stop,
        )

        register_signal.assert_any_call(
            signal.SIGTERM,
            manager.stop,
        )

    def test_wait_returns_false_before_stop(
        self,
        manager: shutdown_manager.ShutdownManager,
    ) -> None:

        assert manager.wait(timeout=0) is False

    def test_stop_causes_wait_to_return_true(
        self,
        manager: shutdown_manager.ShutdownManager,
    ) -> None:

        manager.stop()

        assert manager.wait(timeout=0) is True
