from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class UnsafeControlError(ValueError):
    """Raised when planner outputs are not safe to serialize and publish."""


# sbmpc exposes a measured-state Jacobian du/dx. LFC applies gains to
# (desired - measured), so the bridge must negate the planner gain by default.
SBMPC_TO_LFC_GAIN_SCALE = -1.0


@dataclass(slots=True)
class PlanningDeadlineMonitor:
    """Counts planner-deadline misses for diagnostics (fail-open by default)."""

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


def validate_planner_output(
    tau_ff: np.ndarray,
    feedback_gain: np.ndarray,
    *,
    control_dim: int = 7,
    state_dim: int = 14,
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

    return tau, gain


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
