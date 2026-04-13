from __future__ import annotations

from rclpy.executors import ExternalShutdownException

from sbmpc_ros_bridge import lfc_bridge_node


class FakeNode:
    def __init__(self) -> None:
        self.context = object()
        self.destroyed = False

    def destroy_node(self) -> None:
        self.destroyed = True


def test_main_shuts_down_context_after_a_clean_spin(monkeypatch) -> None:
    fake_node = FakeNode()
    shutdown_contexts: list[object] = []

    monkeypatch.setattr(lfc_bridge_node.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(lfc_bridge_node, "SbMpcLfcBridgeNode", lambda: fake_node)
    monkeypatch.setattr(lfc_bridge_node.rclpy, "spin", lambda node: None)
    monkeypatch.setattr(lfc_bridge_node.rclpy, "ok", lambda context=None: True)
    monkeypatch.setattr(
        lfc_bridge_node.rclpy,
        "shutdown",
        lambda context=None: shutdown_contexts.append(context),
    )

    lfc_bridge_node.main()

    assert fake_node.destroyed is True
    assert shutdown_contexts == [fake_node.context]


def test_main_avoids_double_shutdown_when_context_is_already_closed(
    monkeypatch,
) -> None:
    fake_node = FakeNode()
    shutdown_contexts: list[object] = []

    monkeypatch.setattr(lfc_bridge_node.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(lfc_bridge_node, "SbMpcLfcBridgeNode", lambda: fake_node)

    def fake_spin(node) -> None:
        assert node is fake_node
        raise ExternalShutdownException()

    monkeypatch.setattr(lfc_bridge_node.rclpy, "spin", fake_spin)
    monkeypatch.setattr(lfc_bridge_node.rclpy, "ok", lambda context=None: False)
    monkeypatch.setattr(
        lfc_bridge_node.rclpy,
        "shutdown",
        lambda context=None: shutdown_contexts.append(context),
    )

    lfc_bridge_node.main()

    assert fake_node.destroyed is True
    assert shutdown_contexts == []
