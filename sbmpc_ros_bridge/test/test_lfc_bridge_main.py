from __future__ import annotations

from rclpy.executors import ExternalShutdownException

from sbmpc_ros_bridge import lfc_bridge_node


class FakeNode:
    def __init__(self) -> None:
        self.context = object()
        self.destroyed = False

    def destroy_node(self) -> None:
        self.destroyed = True


class FakeExecutor:
    def __init__(self, *, num_threads: int) -> None:
        self.num_threads = num_threads
        self.nodes = []
        self.shutdown_called = False

    def add_node(self, node) -> None:
        self.nodes.append(node)

    def spin(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_main_shuts_down_context_after_a_clean_spin(monkeypatch) -> None:
    fake_node = FakeNode()
    shutdown_contexts: list[object] = []
    executors: list[FakeExecutor] = []

    monkeypatch.setattr(lfc_bridge_node.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(lfc_bridge_node, "SbMpcLfcBridgeNode", lambda: fake_node)
    monkeypatch.setattr(
        lfc_bridge_node,
        "MultiThreadedExecutor",
        lambda num_threads: executors.append(FakeExecutor(num_threads=num_threads))
        or executors[-1],
    )
    monkeypatch.setattr(lfc_bridge_node.rclpy, "ok", lambda context=None: True)
    monkeypatch.setattr(
        lfc_bridge_node.rclpy,
        "shutdown",
        lambda context=None: shutdown_contexts.append(context),
    )

    lfc_bridge_node.main()

    assert executors[0].nodes == [fake_node]
    assert executors[0].shutdown_called is True
    assert fake_node.destroyed is True
    assert shutdown_contexts == [fake_node.context]


def test_main_avoids_double_shutdown_when_context_is_already_closed(
    monkeypatch,
) -> None:
    fake_node = FakeNode()
    shutdown_contexts: list[object] = []
    executors: list[FakeExecutor] = []

    monkeypatch.setattr(lfc_bridge_node.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(lfc_bridge_node, "SbMpcLfcBridgeNode", lambda: fake_node)

    class RaisingExecutor(FakeExecutor):
        def spin(self) -> None:
            assert self.nodes == [fake_node]
            raise ExternalShutdownException()

    def make_executor(num_threads: int):
        executor = RaisingExecutor(num_threads=num_threads)
        executors.append(executor)
        return executor

    monkeypatch.setattr(lfc_bridge_node, "MultiThreadedExecutor", make_executor)
    monkeypatch.setattr(lfc_bridge_node.rclpy, "ok", lambda context=None: False)
    monkeypatch.setattr(
        lfc_bridge_node.rclpy,
        "shutdown",
        lambda context=None: shutdown_contexts.append(context),
    )

    lfc_bridge_node.main()

    assert executors[0].shutdown_called is True
    assert fake_node.destroyed is True
    assert shutdown_contexts == []


def test_main_converts_sigterm_to_clean_shutdown(monkeypatch) -> None:
    fake_node = FakeNode()
    shutdown_contexts: list[object] = []
    executors: list[FakeExecutor] = []
    installed_handlers = []

    monkeypatch.setattr(lfc_bridge_node.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(lfc_bridge_node, "SbMpcLfcBridgeNode", lambda: fake_node)
    monkeypatch.setattr(
        lfc_bridge_node.signal,
        "getsignal",
        lambda signum: "previous-sigterm-handler",
    )

    def set_signal(signum, handler):
        installed_handlers.append((signum, handler))

    monkeypatch.setattr(lfc_bridge_node.signal, "signal", set_signal)

    class SigtermExecutor(FakeExecutor):
        def spin(self) -> None:
            assert self.nodes == [fake_node]
            signum, handler = installed_handlers[0]
            assert signum == lfc_bridge_node.signal.SIGTERM
            handler(signum, None)

    def make_executor(num_threads: int):
        executor = SigtermExecutor(num_threads=num_threads)
        executors.append(executor)
        return executor

    monkeypatch.setattr(lfc_bridge_node, "MultiThreadedExecutor", make_executor)
    monkeypatch.setattr(lfc_bridge_node.rclpy, "ok", lambda context=None: True)
    monkeypatch.setattr(
        lfc_bridge_node.rclpy,
        "shutdown",
        lambda context=None: shutdown_contexts.append(context),
    )

    lfc_bridge_node.main()

    assert executors[0].shutdown_called is True
    assert fake_node.destroyed is True
    assert shutdown_contexts == [fake_node.context]
    assert installed_handlers[-1] == (
        lfc_bridge_node.signal.SIGTERM,
        "previous-sigterm-handler",
    )
