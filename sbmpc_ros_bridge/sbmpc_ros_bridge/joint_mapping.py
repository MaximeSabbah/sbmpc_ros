from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from linear_feedback_controller_msgs.msg import Sensor
from sensor_msgs.msg import JointState


PANDA_ARM_JOINT_NAMES: tuple[str, ...] = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)


class JointMappingError(ValueError):
    """Raised when an incoming joint-state message does not match expectations."""


@dataclass(frozen=True, slots=True)
class JointMapper:
    """Validate and optionally reorder joint-state messages into planner order."""

    expected_names: tuple[str, ...] = PANDA_ARM_JOINT_NAMES
    aliases: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def panda(cls) -> "JointMapper":
        return cls()

    def reorder_joint_state(
        self,
        joint_state: JointState,
        *,
        allow_reordering: bool = False,
    ) -> JointState:
        canonical_names = tuple(self.aliases.get(name, name) for name in joint_state.name)
        if len(canonical_names) != len(set(canonical_names)):
            raise JointMappingError(
                f"Duplicate joint names are not allowed: {canonical_names!r}."
            )

        if allow_reordering:
            indices = self._indices_for_reordering(canonical_names)
        else:
            indices = self._indices_for_strict_order(canonical_names)

        position = self._select_vector(
            values=joint_state.position,
            field_name="position",
            names=canonical_names,
            indices=indices,
        )
        velocity = self._select_vector(
            values=joint_state.velocity,
            field_name="velocity",
            names=canonical_names,
            indices=indices,
        )
        effort = self._select_optional_vector(
            values=joint_state.effort,
            field_name="effort",
            names=canonical_names,
            indices=indices,
        )

        return JointState(
            header=joint_state.header,
            name=list(self.expected_names),
            position=position.tolist(),
            velocity=velocity.tolist(),
            effort=effort.tolist(),
        )

    def reorder_sensor(
        self,
        sensor: Sensor,
        *,
        allow_reordering: bool = False,
    ) -> Sensor:
        return Sensor(
            header=sensor.header,
            base_pose=sensor.base_pose,
            base_twist=sensor.base_twist,
            joint_state=self.reorder_joint_state(
                sensor.joint_state,
                allow_reordering=allow_reordering,
            ),
            contacts=list(sensor.contacts),
        )

    def _indices_for_strict_order(self, canonical_names: tuple[str, ...]) -> list[int]:
        if canonical_names != self.expected_names:
            self._raise_name_mismatch(canonical_names)
        return list(range(len(self.expected_names)))

    def _indices_for_reordering(self, canonical_names: tuple[str, ...]) -> list[int]:
        incoming = set(canonical_names)
        expected = set(self.expected_names)
        if incoming != expected:
            self._raise_name_mismatch(canonical_names)
        index_by_name = {name: idx for idx, name in enumerate(canonical_names)}
        return [index_by_name[name] for name in self.expected_names]

    def _raise_name_mismatch(self, canonical_names: tuple[str, ...]) -> None:
        expected = self.expected_names
        expected_set = set(expected)
        incoming_set = set(canonical_names)
        missing = [name for name in expected if name not in incoming_set]
        extra = [name for name in canonical_names if name not in expected_set]
        order_only = not missing and not extra and canonical_names != expected
        details: list[str] = [
            f"expected order {expected!r}",
            f"received {canonical_names!r}",
        ]
        if missing:
            details.append(f"missing joints {missing!r}")
        if extra:
            details.append(f"extra joints {extra!r}")
        if order_only:
            details.append(
                "joint names contain the right set but the order is shuffled; "
                "set allow_reordering=True to accept an explicit remap"
            )
        raise JointMappingError("; ".join(details) + ".")

    def _select_vector(
        self,
        *,
        values: list[float] | tuple[float, ...],
        field_name: str,
        names: tuple[str, ...],
        indices: list[int],
    ) -> np.ndarray:
        expected_size = len(names)
        if len(values) != expected_size:
            raise JointMappingError(
                f"joint_state.{field_name} must have length {expected_size}, "
                f"got {len(values)}."
            )
        array = np.asarray(values, dtype=np.float64)
        return array[indices]

    def _select_optional_vector(
        self,
        *,
        values: list[float] | tuple[float, ...],
        field_name: str,
        names: tuple[str, ...],
        indices: list[int],
    ) -> np.ndarray:
        if len(values) == 0:
            return np.asarray([], dtype=np.float64)
        return self._select_vector(
            values=values,
            field_name=field_name,
            names=names,
            indices=indices,
        )
