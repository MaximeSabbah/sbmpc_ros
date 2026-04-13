from __future__ import annotations

import pytest
from linear_feedback_controller_msgs.msg import Sensor
from sensor_msgs.msg import JointState

from sbmpc_ros_bridge.joint_mapping import (
    JointMapper,
    JointMappingError,
    PANDA_ARM_JOINT_NAMES,
)


def make_sensor(
    names: tuple[str, ...] = PANDA_ARM_JOINT_NAMES,
    *,
    positions: list[float] | None = None,
    velocities: list[float] | None = None,
    efforts: list[float] | None = None,
) -> Sensor:
    count = len(names)
    return Sensor(
        joint_state=JointState(
            name=list(names),
            position=positions or [float(i) for i in range(count)],
            velocity=velocities or [float(i) + 0.5 for i in range(count)],
            effort=efforts or [float(i) + 1.0 for i in range(count)],
        )
    )


def test_joint_mapper_accepts_canonical_panda_order() -> None:
    reordered = JointMapper.panda().reorder_joint_state(make_sensor().joint_state)

    assert tuple(reordered.name) == PANDA_ARM_JOINT_NAMES
    assert list(reordered.position) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert list(reordered.velocity) == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]


def test_joint_mapper_rejects_shuffled_joint_names_by_default() -> None:
    names = (
        "panda_joint2",
        "panda_joint1",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )

    with pytest.raises(JointMappingError, match="order is shuffled"):
        JointMapper.panda().reorder_joint_state(make_sensor(names).joint_state)


def test_joint_mapper_can_explicitly_reorder_a_shuffled_sensor() -> None:
    names = (
        "panda_joint3",
        "panda_joint1",
        "panda_joint2",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    sensor = make_sensor(names, positions=[30.0, 10.0, 20.0, 40.0, 50.0, 60.0, 70.0])

    reordered = JointMapper.panda().reorder_joint_state(
        sensor.joint_state,
        allow_reordering=True,
    )

    assert tuple(reordered.name) == PANDA_ARM_JOINT_NAMES
    assert list(reordered.position) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (
            PANDA_ARM_JOINT_NAMES[:-1],
            "missing joints",
        ),
        (
            PANDA_ARM_JOINT_NAMES + ("unexpected_joint",),
            "extra joints",
        ),
    ],
)
def test_joint_mapper_rejects_missing_or_extra_joint_names(
    names: tuple[str, ...],
    message: str,
) -> None:
    sensor = make_sensor(names)

    with pytest.raises(JointMappingError, match=message):
        JointMapper.panda().reorder_joint_state(sensor.joint_state, allow_reordering=True)


def test_joint_mapper_rejects_wrong_vector_lengths() -> None:
    sensor = make_sensor()
    sensor.joint_state.velocity = [0.0, 1.0]

    with pytest.raises(JointMappingError, match="joint_state.velocity"):
        JointMapper.panda().reorder_joint_state(sensor.joint_state)
