from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class UnsafeControlError(ValueError):
    """Raised when planner outputs are not safe to serialize and publish."""


@dataclass(frozen=True, slots=True)
class ControlSafetyLimits:
    max_abs_torque: float | None = None
    max_gain_norm: float | None = None


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
        if float(np.max(np.abs(tau))) > limits.max_abs_torque:
            raise UnsafeControlError(
                f"feedforward exceeds max_abs_torque={limits.max_abs_torque}."
            )
    if limits is not None and limits.max_gain_norm is not None:
        if float(np.linalg.norm(gain)) > limits.max_gain_norm:
            raise UnsafeControlError(
                f"feedback_gain exceeds max_gain_norm={limits.max_gain_norm}."
            )

    return tau, gain


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
