from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from multiprocessing.queues import Queue

import pytest

from sttx import model
from sttx.model import BundleMessage


@pytest.mark.parametrize(
    "message, raises_error",
    [
        (("ok", ("encoder", "decoder", "joiner", "tokens", "silero", "cache")), False),
        (("error", "acquisition failed"), True),
    ],
)
@pytest.mark.parametrize("graceful", [True, False])
def test_completed_worker_can_finish_before_forced_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    message: BundleMessage,
    raises_error: bool,
    graceful: bool,
) -> None:
    # Given: a worker that has sent its result but is still shutting down.
    class Worker:
        alive: bool = True
        terminated: bool = False
        exitcode: int = 0

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            del timeout
            if graceful or self.terminated:
                self.alive = False

        def terminate(self) -> None:
            self.terminated = True

    class Messages:
        def get(self, timeout: float) -> BundleMessage:
            del timeout
            return message

        def close(self) -> None:
            pass

    worker = Worker()
    messages = Messages()

    class Context:
        def Queue(self) -> Messages:
            return messages

        def Process(
            self,
            *,
            target: Callable[[Queue[BundleMessage], str], None],
            args: tuple[Messages, str],
        ) -> Worker:
            del target, args
            return worker

    def context(method: str) -> Context:
        del method
        return Context()

    monkeypatch.setattr(multiprocessing, "get_context", context)

    # When: the parent consumes either a successful bundle or an error result.
    if raises_error:
        with pytest.raises(model.ModelEnvironmentError, match="acquisition failed"):
            _ = model.resolve_bundle_cancellable()
    else:
        bundle = model.resolve_bundle_cancellable()
        assert bundle.source == "cache"

    # Then: only a worker that cannot finish naturally needs SIGTERM.
    assert not worker.alive
    assert worker.terminated is not graceful
