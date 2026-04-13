from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


class UnsafeControlError(ValueError):
    """Raised when planner outputs are not safe to serialize and publish."""


# sbmpc exposes a measured-state Jacobian du/dx. LFC applies gains to
# (desired - measured), so the bridge must negate the planner gain by default.
SBMPC_TO_LFC_GAIN_SCALE = -1.0
TorqueLimitMode = Literal["reject", "clip"]
GainLimitMode = Literal["reject", "scale"]


@dataclass(frozen=True, slots=True)
class AlwaysOnSafety:
    """Checks that should remain enabled in every deployment profile."""

    gain_scale: float = SBMPC_TO_LFC_GAIN_SCALE
    max_control_age_sec: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.gain_scale):
            raise ValueError(f"gain_scale must be finite, got {self.gain_scale}.")
        if self.max_control_age_sec is not None and self.max_control_age_sec < 0.0:
            raise ValueError("max_control_age_sec must be non-negative.")

    def validate_control_age(
        self,
        *,
        control_stamp_sec: float,
        now_sec: float,
    ) -> float:
        if self.max_control_age_sec is None:
            return compute_control_age_sec(
                control_stamp_sec=control_stamp_sec,
                now_sec=now_sec,
            )
        return validate_control_age(
            control_stamp_sec=control_stamp_sec,
            now_sec=now_sec,
            max_control_age_sec=self.max_control_age_sec,
        )


@dataclass(frozen=True, slots=True)
class ControlSafetyLimits:
    max_abs_torque: float | None = None
    max_gain_norm: float | None = None
    max_control_age_sec: float | None = None
    torque_limit_mode: TorqueLimitMode = "reject"
    gain_limit_mode: GainLimitMode = "reject"

    def __post_init__(self) -> None:
        if self.max_abs_torque is not None and self.max_abs_torque < 0.0:
            raise ValueError("max_abs_torque must be non-negative.")
        if self.max_gain_norm is not None and self.max_gain_norm < 0.0:
            raise ValueError("max_gain_norm must be non-negative.")
        if self.max_control_age_sec is not None and self.max_control_age_sec < 0.0:
            raise ValueError("max_control_age_sec must be non-negative.")
        if self.torque_limit_mode not in ("reject", "clip"):
            raise ValueError(
                "torque_limit_mode must be either 'reject' or 'clip'."
            )
        if self.gain_limit_mode not in ("reject", "scale"):
            raise ValueError(
                "gain_limit_mode must be either 'reject' or 'scale'."
            )


@dataclass(frozen=True, slots=True)
class BringupLimits:
    """Optional conservative limits used during early robot bringup."""

    max_abs_torque: float | None = None
    max_gain_norm: float | None = None
    torque_limit_mode: TorqueLimitMode = "reject"
    gain_limit_mode: GainLimitMode = "reject"

    def __post_init__(self) -> None:
        if self.max_abs_torque is not None and self.max_abs_torque < 0.0:
            raise ValueError("max_abs_torque must be non-negative.")
        if self.max_gain_norm is not None and self.max_gain_norm < 0.0:
            raise ValueError("max_gain_norm must be non-negative.")
        if self.torque_limit_mode not in ("reject", "clip"):
            raise ValueError(
                "torque_limit_mode must be either 'reject' or 'clip'."
            )
        if self.gain_limit_mode not in ("reject", "scale"):
            raise ValueError(
                "gain_limit_mode must be either 'reject' or 'scale'."
            )

    def as_control_safety_limits(self) -> ControlSafetyLimits:
        return ControlSafetyLimits(
            max_abs_torque=self.max_abs_torque,
            max_gain_norm=self.max_gain_norm,
            torque_limit_mode=self.torque_limit_mode,
            gain_limit_mode=self.gain_limit_mode,
        )


@dataclass(slots=True)
class PlanningDeadlineMonitor:
    max_planning_duration_sec: float
    fail_closed: bool = True
    deadline_miss_count: int = 0
    last_planning_duration_sec: float | None = None

    def __post_init__(self) -> None:
        if self.max_planning_duration_sec < 0.0:
            raise ValueError("max_planning_duration_sec must be non-negative.")

    def observe(self, planning_duration_sec: float) -> bool:
        duration = float(planning_duration_sec)
        if not np.isfinite(duration):
            raise UnsafeControlError("planning_duration_sec must be finite.")
        if duration < 0.0:
            raise UnsafeControlError("planning_duration_sec must be non-negative.")

        self.last_planning_duration_sec = duration
        deadline_met = duration <= self.max_planning_duration_sec
        if deadline_met:
            return True

        self.deadline_miss_count += 1
        if self.fail_closed:
            raise UnsafeControlError(
                "planner deadline missed: "
                f"{duration:.6f}s > {self.max_planning_duration_sec:.6f}s."
            )
        return False


@dataclass(frozen=True, slots=True)
class MonitoringOnly:
    """Signals worth observing even when they should not limit motion by default."""

    max_planning_duration_sec: float | None = None
    fail_closed_on_deadline_miss: bool = False

    def __post_init__(self) -> None:
        if (
            self.max_planning_duration_sec is not None
            and self.max_planning_duration_sec < 0.0
        ):
            raise ValueError("max_planning_duration_sec must be non-negative.")

    def make_deadline_monitor(self) -> PlanningDeadlineMonitor | None:
        if self.max_planning_duration_sec is None:
            return None
        return PlanningDeadlineMonitor(
            max_planning_duration_sec=self.max_planning_duration_sec,
            fail_closed=self.fail_closed_on_deadline_miss,
        )


@dataclass(frozen=True, slots=True)
class BridgeSafetyProfile:
    """High-level safety split for the ROS bridge runtime."""

    always_on: AlwaysOnSafety = field(default_factory=AlwaysOnSafety)
    bringup_limits: BringupLimits = field(default_factory=BringupLimits)
    monitoring_only: MonitoringOnly = field(default_factory=MonitoringOnly)

    def planner_output_limits(self) -> ControlSafetyLimits:
        return self.bringup_limits.as_control_safety_limits()

    def make_deadline_monitor(self) -> PlanningDeadlineMonitor | None:
        return self.monitoring_only.make_deadline_monitor()


def make_default_safety_profile() -> BridgeSafetyProfile:
    """Default runtime profile: always-on checks, no extra clipping, monitoring only."""

    return BridgeSafetyProfile()


def make_conservative_bringup_profile(
    *,
    max_control_age_sec: float | None = None,
    max_planning_duration_sec: float | None = None,
    max_abs_torque: float | None = None,
    max_gain_norm: float | None = None,
    torque_limit_mode: TorqueLimitMode = "clip",
    gain_limit_mode: GainLimitMode = "scale",
    fail_closed_on_deadline_miss: bool = False,
) -> BridgeSafetyProfile:
    return BridgeSafetyProfile(
        always_on=AlwaysOnSafety(max_control_age_sec=max_control_age_sec),
        bringup_limits=BringupLimits(
            max_abs_torque=max_abs_torque,
            max_gain_norm=max_gain_norm,
            torque_limit_mode=torque_limit_mode,
            gain_limit_mode=gain_limit_mode,
        ),
        monitoring_only=MonitoringOnly(
            max_planning_duration_sec=max_planning_duration_sec,
            fail_closed_on_deadline_miss=fail_closed_on_deadline_miss,
        ),
    )


def validate_planner_output(
    tau_ff: np.ndarray,
    feedback_gain: np.ndarray,
    *,
    control_dim: int = 7,
    state_dim: int = 14,
    limits: ControlSafetyLimits | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau_ff, dtype=np.float64)
    gain = np.asarray(feedback_gain, dtype=np.float64)

    if tau.shape != (control_dim,):
        raise UnsafeControlError(
            f"feedforward must have shape ({control_dim},), got {tau.shape}."
        )
    if gain.shape != (control_dim, state_dim):
        raise UnsafeControlError(
            "feedback_gain must have shape "
            f"({control_dim}, {state_dim}), got {gain.shape}."
        )
    if not np.all(np.isfinite(tau)):
        raise UnsafeControlError("feedforward contains non-finite values.")
    if not np.all(np.isfinite(gain)):
        raise UnsafeControlError("feedback_gain contains non-finite values.")

    if limits is not None and limits.max_abs_torque is not None:
        tau = apply_torque_limit(
            tau,
            max_abs_torque=limits.max_abs_torque,
            mode=limits.torque_limit_mode,
        )
    if limits is not None and limits.max_gain_norm is not None:
        gain = apply_gain_norm_limit(
            gain,
            max_gain_norm=limits.max_gain_norm,
            mode=limits.gain_limit_mode,
        )

    return tau, gain


def apply_torque_limit(
    tau_ff: np.ndarray,
    *,
    max_abs_torque: float,
    mode: TorqueLimitMode = "reject",
) -> np.ndarray:
    tau = np.asarray(tau_ff, dtype=np.float64)
    if max_abs_torque < 0.0:
        raise ValueError("max_abs_torque must be non-negative.")

    peak = float(np.max(np.abs(tau), initial=0.0))
    if peak <= max_abs_torque:
        return tau
    if mode == "clip":
        return np.clip(tau, -max_abs_torque, max_abs_torque)
    if mode == "reject":
        raise UnsafeControlError(
            f"feedforward exceeds max_abs_torque={max_abs_torque}."
        )
    raise ValueError(f"unsupported torque limit mode: {mode!r}.")


def apply_gain_norm_limit(
    feedback_gain: np.ndarray,
    *,
    max_gain_norm: float,
    mode: GainLimitMode = "reject",
) -> np.ndarray:
    gain = np.asarray(feedback_gain, dtype=np.float64)
    if max_gain_norm < 0.0:
        raise ValueError("max_gain_norm must be non-negative.")

    norm = float(np.linalg.norm(gain))
    if norm <= max_gain_norm:
        return gain
    if mode == "scale":
        if norm == 0.0:
            return gain
        return gain * (max_gain_norm / norm)
    if mode == "reject":
        raise UnsafeControlError(
            f"feedback_gain exceeds max_gain_norm={max_gain_norm}."
        )
    raise ValueError(f"unsupported gain limit mode: {mode!r}.")


def sbmpc_gain_to_lfc_gain(
    feedback_gain: np.ndarray,
    *,
    gain_scale: float = SBMPC_TO_LFC_GAIN_SCALE,
) -> np.ndarray:
    if not np.isfinite(gain_scale):
        raise ValueError(f"gain_scale must be finite, got {gain_scale}.")

    gain = np.asarray(feedback_gain, dtype=np.float64)
    if not np.all(np.isfinite(gain)):
        raise UnsafeControlError("feedback_gain contains non-finite values.")
    return gain * float(gain_scale)


def compute_lfc_state_error(
    desired_state: np.ndarray,
    measured_state: np.ndarray,
) -> np.ndarray:
    desired = np.asarray(desired_state, dtype=np.float64)
    measured = np.asarray(measured_state, dtype=np.float64)
    if desired.ndim != 1 or measured.ndim != 1:
        raise ValueError("desired_state and measured_state must be 1D vectors.")
    if desired.shape != measured.shape:
        raise ValueError(
            f"desired_state shape {desired.shape} does not match "
            f"measured_state shape {measured.shape}."
        )
    return desired - measured


def compute_lfc_control(
    feedforward: np.ndarray,
    feedback_gain: np.ndarray,
    desired_state: np.ndarray,
    measured_state: np.ndarray,
) -> np.ndarray:
    tau = np.asarray(feedforward, dtype=np.float64)
    gain = np.asarray(feedback_gain, dtype=np.float64)
    diff_state = compute_lfc_state_error(desired_state, measured_state)

    if tau.ndim != 1:
        raise ValueError(f"feedforward must be 1D, got {tau.shape}.")
    if gain.shape != (tau.size, diff_state.size):
        raise ValueError(
            "feedback_gain shape must match "
            f"({tau.size}, {diff_state.size}), got {gain.shape}."
        )
    if not np.all(np.isfinite(tau)) or not np.all(np.isfinite(gain)):
        raise UnsafeControlError("LFC control inputs must be finite.")

    return tau + gain @ diff_state


def compute_control_age_sec(
    *,
    control_stamp_sec: float,
    now_sec: float,
) -> float:
    stamp = float(control_stamp_sec)
    now = float(now_sec)
    if not np.isfinite(stamp) or not np.isfinite(now):
        raise UnsafeControlError("control timestamps must be finite.")
    age = now - stamp
    if age < 0.0:
        raise UnsafeControlError(
            f"control timestamp is in the future: age_sec={age:.6f}."
        )
    return age


def validate_control_age(
    *,
    control_stamp_sec: float,
    now_sec: float,
    max_control_age_sec: float,
) -> float:
    if max_control_age_sec < 0.0:
        raise ValueError("max_control_age_sec must be non-negative.")
    age = compute_control_age_sec(
        control_stamp_sec=control_stamp_sec,
        now_sec=now_sec,
    )
    if age > max_control_age_sec:
        raise UnsafeControlError(
            f"control is stale: age_sec={age:.6f} exceeds "
            f"max_control_age_sec={max_control_age_sec:.6f}."
        )
    return age
