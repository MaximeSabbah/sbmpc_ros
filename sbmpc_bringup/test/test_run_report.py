from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sbmpc_bringup.run_report import (
    ControlSeries,
    DiagnosticSeries,
    JointSeries,
    RunData,
    _arm_indices,
    _controller_activity_rows,
    _diagnostic_statistics,
    _gripper_joint_row,
    _joint_row,
    _plot_agimus_ee_wrench,
    _plot_agimus_joint_motor,
    _plot_agimus_robot_state,
    _plot_controller,
    _plot_phase,
    _plot_task,
    _plot_mujoco_actuators,
    _plot_simulation_ground_truth,
    _reconstruct_agimus_limiter,
    _resolve_bag,
    _robot_state_row,
    _ros_diagnostic_rows,
    _write_agimus_robot_state_csv,
    _write_controller_activity_csv,
    _write_controller_diagnostics_csv,
    _write_steps_csv,
    _write_simulation_csvs,
    _windowed_clock_rtf,
    align_run,
    causal_indices,
    nearest_indices,
    summarize,
)


def joint_series(times: list[float], q: np.ndarray) -> JointSeries:
    count = len(times)
    return JointSeries(
        receive=np.asarray(times),
        stamp=np.asarray(times),
        q=np.asarray(q, dtype=float),
        v=np.zeros((count, 7)),
        effort=np.zeros((count, 7)),
    )


def synthetic_run(*, phase_machine: bool = False) -> RunData:
    times = np.asarray([10.0, 10.04, 10.08])
    q = np.zeros((3, 7))
    q[:, 0] = [0.1, 0.05, 0.02]
    gain = np.zeros((3, 7, 14))
    gain[:, 0, 0] = 2.0
    diagnostics = []
    for index, error in enumerate((0.1, 0.05, 0.02), start=1):
        orientation_error = 0.04 - 0.01 * index
        cosine = np.cos(orientation_error)
        sine = np.sin(orientation_error)
        goal_rotation = np.diag([1.0, -1.0, -1.0])
        ee_rotation = goal_rotation @ np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        phase = None
        if phase_machine:
            phase = {
                "phase": "PREGRASP",
                "plan_time_sec": 0.04 * (index - 1),
                "at_boundary": index == 3,
                "transition_blocked": index == 3,
                "q_ok": False,
                "velocity_ok": True,
                "ee_position_error_norm_m": error,
            }
        diagnostics.append(
            {
                "planner_step_count": index,
                "planner_mode": "exact_feedback",
                "last_phase": "PREGRASP",
                "last_reference_q": [0.0] * 7,
                "last_reference_v": [0.0] * 7,
                "last_goal_position": [0.5, 0.0, 0.18],
                "last_position_error": error,
                "last_position_error_signed": [-error, 0.0, 0.0],
                "last_orientation_error": orientation_error,
                "last_ee_position": [0.5 - error, 0.0, 0.18],
                "last_ee_rotation": ee_rotation.reshape(-1).tolist(),
                "last_goal_rotation": goal_rotation.reshape(-1).tolist(),
                "last_gain_ess": 32.0 * index,
                "last_gain_nominal_weight": 0.4 - 0.1 * index,
                "last_planning_time_ms": 25.0,
                "last_planner_step_wall_time_ms": 26.0,
                "accepted_planner_output_count": index,
                "published_control_count": index,
                "rejected_planner_output_count": 0,
                "deadline_miss_count": 0,
                "rejected_sensor_count": 0,
                "last_error": "",
                "phase_machine": phase,
                "gripper": {"stage": "idle", "goal_count": 0},
            }
        )
    state = joint_series(times.tolist(), q)
    return RunData(
        bag=Path("/tmp/example/rosbag"),
        control=ControlSeries(
            receive=times,
            stamp=times,
            anchor_stamp=times - 0.04,
            feedforward=np.ones((3, 7)),
            gain=gain,
            anchor_q=np.zeros((3, 7)),
            anchor_v=np.zeros((3, 7)),
        ),
        sensor=state,
        output=joint_series(times.tolist(), q),
        hardware=joint_series(times.tolist(), np.zeros((3, 7))),
        merged=None,
        desired=None,
        robot_state=[],
        diagnostics=DiagnosticSeries(receive=times, rows=diagnostics),
        rosout=[],
        ros_diagnostics=[],
        controller_activity=[],
        gripper_joint_states=[],
        gripper_feedback=[],
        gripper_status=[],
        topic_counts={
            "/control": 3,
            "/sensor": 3,
            "/output_joint_effort": 3,
            "/franka/joint_states": 3,
            "/franka_robot_state_broadcaster/robot_state": 0,
            "/franka_robot_state_broadcaster/desired_joint_states": 0,
        },
    )


class BooleanFields(SimpleNamespace):
    def get_fields_and_field_types(self) -> dict[str, str]:
        return {name: "boolean" for name in vars(self)}


def stamp(seconds: float) -> SimpleNamespace:
    whole = int(seconds)
    return SimpleNamespace(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))


def header(seconds: float = 2.0, frame_id: str = "fer_link0") -> SimpleNamespace:
    return SimpleNamespace(stamp=stamp(seconds), frame_id=frame_id)


def vector(values: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SimpleNamespace:
    return SimpleNamespace(x=values[0], y=values[1], z=values[2])


def pose_stamped(position=(0.5, 0.0, 0.2), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        header=header(),
        pose=SimpleNamespace(
            position=vector(position),
            orientation=SimpleNamespace(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        ),
    )


def spatial_stamped(kind: str, first=(0.0, 0.0, 0.0), second=(0.0, 0.0, 0.0)):
    spatial = SimpleNamespace()
    if kind == "wrench":
        spatial.force = vector(first)
        spatial.torque = vector(second)
    else:
        spatial.linear = vector(first)
        spatial.angular = vector(second)
    return SimpleNamespace(header=header(), **{kind: spatial})


def inertia_stamped(mass: float) -> SimpleNamespace:
    return SimpleNamespace(
        header=header(),
        inertia=SimpleNamespace(
            m=mass,
            com=vector((0.01, 0.02, 0.03)),
            ixx=1.0,
            ixy=0.1,
            ixz=0.2,
            iyy=2.0,
            iyz=0.3,
            izz=3.0,
        ),
    )


def joint_state(
    *,
    names: list[str] | None = None,
    position: list[float] | None = None,
    velocity: list[float] | None = None,
    effort: list[float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        header=header(),
        name=names or [f"fer_joint{i}" for i in range(1, 8)],
        position=[0.0] * 7 if position is None else position,
        velocity=[0.0] * 7 if velocity is None else velocity,
        effort=[0.0] * 7 if effort is None else effort,
    )


def agimus_message(*, seconds: float = 2.0, tau_j_d: float = 0.1) -> SimpleNamespace:
    zero = vector()
    measured = joint_state(
        position=[0.01 * index for index in range(7)],
        velocity=[0.001 * index for index in range(7)],
        effort=[0.2 * index for index in range(7)],
    )
    desired = joint_state(
        position=[0.02 * index for index in range(7)],
        velocity=[0.002 * index for index in range(7)],
        effort=[tau_j_d] * 7,
    )
    motor = joint_state(
        position=[0.011 * index for index in range(7)],
        velocity=[0.0011 * index for index in range(7)],
        effort=[],
    )
    external = joint_state(position=[], velocity=[], effort=[0.01] * 7)
    return SimpleNamespace(
        header=header(seconds),
        collision_indicators=SimpleNamespace(
            is_cartesian_linear_collision=zero,
            is_cartesian_angular_collision=zero,
            is_cartesian_linear_contact=vector((1.0, 0.0, 0.0)),
            is_cartesian_angular_contact=zero,
            is_joint_collision=[0.0] * 7,
            is_joint_contact=[0.0] * 7,
        ),
        measured_joint_state=measured,
        desired_joint_state=desired,
        measured_joint_motor_state=motor,
        ddq_d=[0.0] * 7,
        dtau_j=[0.0] * 7,
        tau_ext_hat_filtered=external,
        elbow=SimpleNamespace(
            position=[0.0, 1.0],
            desired_position=[0.0, 1.0],
            commanded_position=[0.0, 1.0],
            commanded_velocity=[0.0, 0.0],
            commanded_acceleration=[0.0, 0.0],
        ),
        k_f_ext_hat_k=spatial_stamped("wrench", (1.0, 2.0, 3.0)),
        o_f_ext_hat_k=spatial_stamped("wrench", (4.0, 5.0, 6.0)),
        inertia_ee=inertia_stamped(0.73),
        inertia_load=inertia_stamped(0.2),
        inertia_total=inertia_stamped(0.93),
        o_t_ee=pose_stamped(),
        o_t_ee_d=pose_stamped((0.5, 0.0, 0.21)),
        o_t_ee_c=pose_stamped((0.5, 0.0, 0.22)),
        f_t_ee=pose_stamped((0.0, 0.0, 0.1)),
        ee_t_k=pose_stamped((0.0, 0.0, 0.0)),
        o_dp_ee_d=spatial_stamped("twist"),
        o_dp_ee_c=spatial_stamped("twist"),
        o_ddp_ee_c=spatial_stamped("accel"),
        time=seconds,
        control_command_success_rate=0.99,
        robot_mode=2,
        current_errors=BooleanFields(joint_velocity_violation=False),
        last_motion_errors=BooleanFields(controller_torque_discontinuity=True),
    )


def test_nearest_indices_selects_closest_rows() -> None:
    source = np.asarray([0.0, 1.0, 2.0])
    query = np.asarray([0.1, 0.6, 1.9])

    assert nearest_indices(source, query).tolist() == [0, 1, 2]


def test_causal_indices_never_select_a_row_before_the_query() -> None:
    source = np.asarray([0.0, 1.0, 2.0])
    query = np.asarray([0.1, 1.0, 1.9])

    assert causal_indices(source, query).tolist() == [1, 1, 2]


def test_sensor_row_uses_outer_sensor_header_and_named_joint_order() -> None:
    names = list(reversed([f"fer_joint{i}" for i in range(1, 8)]))
    joint_state = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
        name=names,
        position=list(range(7)),
        velocity=[0.0] * 7,
        effort=[0.0] * 7,
    )
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=500_000_000)),
        joint_state=joint_state,
    )

    row = _joint_row(message, 13.0, sensor=True)

    assert row["stamp"] == pytest.approx(12.5)
    assert row["q"] == list(reversed(range(7)))


def test_arm_mapping_rejects_partial_or_unknown_names() -> None:
    with pytest.raises(ValueError, match="unambiguous"):
        _arm_indices(["wrong_joint"] * 7, 7)


def test_agimus_robot_state_row_preserves_tau_j_d_mode_and_errors() -> None:
    message = agimus_message(tau_j_d=1.5)
    message.robot_mode = 4
    message.current_errors = BooleanFields(
        joint_velocity_violation=True, joint_reflex=False
    )

    row = _robot_state_row(message, 2.1)

    assert row["desired_effort"] == [1.5] * 7
    assert row["motor_position"][6] == pytest.approx(0.066)
    assert row["external_wrench_base"][:3] == [4.0, 5.0, 6.0]
    assert row["inertia_load"][0] == pytest.approx(0.2)
    assert row["desired_ee_pose"][2] == pytest.approx(0.21)
    assert row["contact_indicator_max"] == pytest.approx(1.0)
    assert row["robot_mode"] == 4
    assert row["current_errors"] == ["joint_velocity_violation"]
    assert row["last_motion_errors"] == ["controller_torque_discontinuity"]


def test_agimus_state_is_summarized_and_plotted_when_available(
    tmp_path: Path,
) -> None:
    robot_row = _robot_state_row(agimus_message(seconds=10.04), 10.041)
    run = replace(synthetic_run(), robot_state=[robot_row])
    aligned = align_run(run)

    summary = summarize(aligned, terminal_sec=1.0)
    filename = _plot_agimus_robot_state(aligned, tmp_path)
    joint_filename = _plot_agimus_joint_motor(aligned, tmp_path)
    ee_filename = _plot_agimus_ee_wrench(aligned, tmp_path)
    _write_agimus_robot_state_csv(aligned, tmp_path)

    assert summary["agimus_robot_state"]["available"] is True
    assert summary["agimus_robot_state"]["robot_mode_counts"] == {"MOVE": 1}
    assert summary["agimus_robot_state"]["last_motion_errors_seen"] == [
        "controller_torque_discontinuity"
    ]
    assert filename == "09_agimus_robot_state.png"
    assert (tmp_path / filename).is_file()
    assert joint_filename == "10_agimus_joint_motor.png"
    assert ee_filename == "11_agimus_ee_wrench.png"
    csv_text = (tmp_path / "agimus_robot_state.csv").read_text()
    assert "measured_position_j1" in csv_text
    assert "inertia_load_mass" in csv_text


def test_limiter_reconstruction_uses_header_causality_not_receive_time() -> None:
    output_stamp = 1.0 + np.arange(11, dtype=float) * 0.001
    state_q = np.asarray(
        [[0.01 * index for index in range(7)]] * len(output_stamp)
    )
    output = JointSeries(
        receive=np.arange(len(output_stamp), dtype=float)[::-1] + 100.0,
        stamp=output_stamp,
        q=state_q,
        v=np.zeros_like(state_q),
        effort=np.full_like(state_q, 2.0),
    )
    first = _robot_state_row(agimus_message(seconds=1.0, tau_j_d=0.0), 50.0)
    second = _robot_state_row(agimus_message(seconds=1.01, tau_j_d=2.0), -50.0)
    data = replace(synthetic_run(), output=output, robot_state=[first, second])

    reconstruction = _reconstruct_agimus_limiter(data)

    assert reconstruction is not None
    assert reconstruction.valid_interval.tolist() == [False, True]
    assert reconstruction.interval_output_count.tolist() == [0, 10]
    assert reconstruction.predicted[1].tolist() == pytest.approx([2.0] * 7)
    assert reconstruction.observed[1].tolist() == pytest.approx([2.0] * 7)


def test_limiter_reconstruction_marks_missing_1khz_output_invalid() -> None:
    output_stamp = np.asarray([1.0, 1.001, 1.002, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.01])
    q = np.asarray([[0.01 * index for index in range(7)]] * len(output_stamp))
    output = JointSeries(
        receive=output_stamp,
        stamp=output_stamp,
        q=q,
        v=np.zeros_like(q),
        effort=np.zeros_like(q),
    )
    states = [
        _robot_state_row(agimus_message(seconds=1.0, tau_j_d=0.0), 1.0),
        _robot_state_row(agimus_message(seconds=1.01, tau_j_d=0.0), 1.01),
    ]

    reconstruction = _reconstruct_agimus_limiter(
        replace(synthetic_run(), output=output, robot_state=states)
    )

    assert reconstruction is not None
    assert not reconstruction.valid_interval[1]
    assert reconstruction.interval_max_gap_ms[1] == pytest.approx(2.0)


def test_ros_diagnostics_and_activity_are_decoded_and_exported(tmp_path: Path) -> None:
    range_value = "Avg: 16.34 [11.79 - 99.31] us, StdDev: 6.88"
    status = SimpleNamespace(
        level=b"\x02",
        name="controller_manager: Hardware Components Activity",
        message="High execution jitter",
        hardware_id="fer",
        values=[
            SimpleNamespace(
                key="AgimusFrankaHardwareInterface.read_cycle.execution_time",
                value=range_value,
            )
        ],
    )
    diagnostic_message = SimpleNamespace(header=header(10.0), status=[status])
    diagnostic_rows = _ros_diagnostic_rows(diagnostic_message, 10.1)
    activity_message = SimpleNamespace(
        header=header(10.0),
        controllers=[
            SimpleNamespace(
                name="linear_feedback_controller",
                state=SimpleNamespace(id=3, label="active"),
            )
        ],
        hardware_components=[
            SimpleNamespace(
                name="AgimusFrankaHardwareInterface",
                state=SimpleNamespace(id=3, label="active"),
            )
        ],
    )
    activity_rows = _controller_activity_rows(activity_message, 10.1)
    run = replace(
        synthetic_run(),
        ros_diagnostics=diagnostic_rows,
        controller_activity=activity_rows,
    )
    aligned = align_run(run)

    parsed = _diagnostic_statistics(range_value)
    _write_controller_diagnostics_csv(aligned, tmp_path)
    _write_controller_activity_csv(aligned, tmp_path)

    assert parsed is not None
    assert parsed["average"] == pytest.approx(16.34)
    assert parsed["maximum"] == pytest.approx(99.31)
    assert diagnostic_rows[0]["level"] == 2
    assert {row["kind"] for row in activity_rows} == {"controller", "hardware"}
    assert "read_cycle.execution_time" in (
        tmp_path / "controller_diagnostics.csv"
    ).read_text()
    assert "linear_feedback_controller" in (
        tmp_path / "controller_activity.csv"
    ).read_text()


def test_resolve_bag_accepts_run_or_direct_bag(tmp_path: Path) -> None:
    bag = tmp_path / "run" / "rosbag"
    bag.mkdir(parents=True)
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}")

    assert _resolve_bag(tmp_path / "run") == bag.resolve()
    assert _resolve_bag(bag) == bag.resolve()


def test_alignment_reconstructs_deployed_lfc_feedback_sign() -> None:
    aligned = align_run(synthetic_run())

    # K_msg=+2 and [anchor-current]=-0.1 => signed correction=-0.2 Nm.
    assert aligned.feedback[0, 0] == pytest.approx(-0.2)
    assert aligned.feedback[-1, 0] == pytest.approx(-0.04)


def test_summary_separates_bias_runtime_and_observability() -> None:
    summary = summarize(align_run(synthetic_run()), terminal_sec=1.0)

    assert summary["recorded_goal_position_m"] == [0.5, 0.0, 0.18]
    assert summary["final_ee_position_m"] == pytest.approx([0.48, 0.0, 0.18])
    assert summary["final_ee_position_error_signed_mm"] == pytest.approx(
        [-20.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        summary["recorded_goal_rotation_matrix"],
        np.diag([1.0, -1.0, -1.0]),
    )
    assert summary["final_ee_rotation_matrix"] is not None
    assert summary["final_q_minus_reference_rad"][0] == pytest.approx(0.02)
    assert summary["final_q_minus_local_anchor_rad"][0] == pytest.approx(0.02)
    assert summary["terminal_position_error_mm"]["median"] == pytest.approx(50.0)
    assert summary["ee_position_error_signed_mm_by_axis"]["x"][
        "median"
    ] == pytest.approx(-50.0)
    assert summary["terminal_ee_position_error_signed_mm_by_axis"]["z"][
        "max"
    ] == pytest.approx(0.0)
    assert summary["orientation_error_rad"]["count"] == 3
    assert summary["gain_sampling"]["ess"]["median"] == pytest.approx(64.0)
    assert summary["gain_sampling"]["nominal_weight"]["median"] == pytest.approx(
        0.2
    )
    assert summary["control_period_ms"]["median"] == pytest.approx(40.0)
    assert summary["solve_anchor_age_ms"]["median"] == pytest.approx(40.0)
    assert summary["observability"]["post_rate_limit_tau_J_d_available"] is False
    assert summary["torque_rate_audit"]["component_steps_over_threshold_total"] == 0
    assert summary["state_stream_consistency"]["all_recorded_state_values_finite"]


def test_summary_prefers_phase_goal_without_losing_task_goal() -> None:
    run = synthetic_run(phase_machine=True)
    for row in run.diagnostics.rows:
        row["phase_machine"]["ee_goal_position_m"] = [0.5, 0.0, 0.18]
        row["phase_machine"]["ee_position_m"] = [0.501, 0.002, 0.183]
        row["phase_machine"]["ee_position_error_signed_m"] = [
            0.001,
            0.002,
            0.003,
        ]
        row["last_goal_position"] = [0.65, 0.0, 0.105]

    summary = summarize(align_run(run), terminal_sec=1.0)

    assert summary["recorded_goal_position_m"] == [0.5, 0.0, 0.18]
    assert summary["recorded_phase_goal_position_m"] == [0.5, 0.0, 0.18]
    assert summary["recorded_task_goal_position_m"] == [0.65, 0.0, 0.105]
    assert summary["final_ee_position_m"] == pytest.approx([0.501, 0.002, 0.183])
    assert summary["final_ee_position_error_signed_mm"] == pytest.approx(
        [1.0, 2.0, 3.0]
    )


def test_task_controller_plots_and_steps_csv_include_pose_and_gain_health(
    tmp_path: Path,
) -> None:
    run = align_run(synthetic_run())

    assert _plot_task(run, tmp_path) == "01_task_tracking.png"
    assert _plot_controller(run, tmp_path) == "04_controller.png"
    _write_steps_csv(run, tmp_path)

    assert (tmp_path / "01_task_tracking.png").is_file()
    assert (tmp_path / "04_controller.png").is_file()
    with (tmp_path / "controller_steps.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[-1]["ee_error_x_mm"]) == pytest.approx(-20.0)
    assert float(rows[-1]["ee_orientation_error_rad"]) == pytest.approx(0.01)
    assert float(rows[-1]["ee_position_x_m"]) == pytest.approx(0.48)
    assert float(rows[-1]["goal_rotation_r22"]) == pytest.approx(-1.0)
    assert float(rows[-1]["gain_ess"]) == pytest.approx(96.0)
    assert float(rows[-1]["gain_nominal_weight"]) == pytest.approx(0.1)


def test_report_remains_compatible_with_diagnostics_recorded_before_pose_fields(
    tmp_path: Path,
) -> None:
    data = synthetic_run()
    new_fields = (
        "last_position_error_signed",
        "last_orientation_error",
        "last_ee_position",
        "last_ee_rotation",
        "last_goal_rotation",
        "last_gain_ess",
        "last_gain_nominal_weight",
    )
    for row in data.diagnostics.rows:
        for field in new_fields:
            row.pop(field)
    run = align_run(data)

    summary = summarize(run, terminal_sec=1.0)
    assert summary["final_ee_position_m"] is None
    assert summary["final_ee_position_error_signed_mm"] is None
    assert summary["orientation_error_rad"]["count"] == 0
    assert summary["gain_sampling"]["ess"]["count"] == 0
    assert _plot_task(run, tmp_path) == "01_task_tracking.png"
    assert _plot_controller(run, tmp_path) == "04_controller.png"


def test_phase_plot_is_automatic_only_when_phase_machine_is_recorded(
    tmp_path: Path,
) -> None:
    assert _plot_phase(align_run(synthetic_run()), tmp_path) is None

    filename = _plot_phase(align_run(synthetic_run(phase_machine=True)), tmp_path)
    assert filename == "07_phase_and_gripper.png"
    assert (tmp_path / filename).is_file()


def test_simulation_summary_and_artifacts_use_simulator_semantics(
    tmp_path: Path,
) -> None:
    base = synthetic_run()
    object_rows = [
        {
            "receive": 10.0 + 0.5 * index,
            "stamp": 1.0 + 0.5 * index,
            "frame_id": "odom",
            "child_frame_id": "object",
            "pose": [0.5 + 0.01 * index, 0.0, 0.105 + 0.02 * index, 0, 0, 0, 1],
            "twist": [0.01, 0.0, 0.02, 0.0, 0.0, 0.0],
        }
        for index in range(3)
    ]
    actuator_rows = [
        {
            "receive": 10.0 + 0.5 * index,
            "stamp": 1.0 + 0.5 * index,
            "names": ["fer_joint1", "fer_finger_joint1"],
            "position": [0.1 * index, 0.04],
            "velocity": [0.0, 0.0],
            "effort": [1.0, 0.0],
        }
        for index in range(3)
    ]
    run = replace(
        base,
        backend="mujoco",
        backend_source="manifest",
        hardware_source="/joint_states",
        merged=base.hardware,
        sim_object_pose=object_rows,
        sim_actuator_states=actuator_rows,
        sim_clock=[
            {"receive": 10.0, "sim_time": 0.0},
            {"receive": 10.5, "sim_time": 0.25},
            {"receive": 11.0, "sim_time": 0.50},
        ],
    )
    aligned = align_run(run)

    summary = summarize(aligned, terminal_sec=1.0)
    simulation_plot = _plot_simulation_ground_truth(aligned, tmp_path)
    actuator_plot = _plot_mujoco_actuators(aligned, tmp_path)
    _write_simulation_csvs(aligned, tmp_path)

    assert summary["backend"] == "mujoco"
    assert summary["torque_rate_audit"]["component_step_audit_threshold_nm"] is None
    assert "MuJoCo" in summary["agimus_robot_state"]["note"]
    assert summary["simulation"]["clock"]["aggregate_real_time_factor"] == pytest.approx(0.5)
    assert "final_xy_error_to_recorded_task_goal_m" not in summary["simulation"]["object"]
    missing = summary["observability"]["unpopulated_expected_topics"]
    assert "/gripper_action_controller/gripper_cmd/_action/feedback" not in missing
    assert "/gripper_action_controller/gripper_cmd/_action/status" not in missing
    assert simulation_plot == "14_simulation_ground_truth.png"
    assert actuator_plot == "15_mujoco_actuators.png"
    assert (tmp_path / "simulation_object_pose.csv").is_file()
    assert (tmp_path / "mujoco_actuator_states.csv").is_file()
    assert (tmp_path / "simulation_clock.csv").is_file()


def test_gripper_width_handles_single_coupled_sim_finger() -> None:
    message = joint_state(
        names=[f"fer_joint{i}" for i in range(1, 8)] + ["fer_finger_joint1"],
        position=[0.0] * 7 + [0.04],
        velocity=[0.0] * 8,
        effort=[0.0] * 8,
    )

    row = _gripper_joint_row(message, 2.1)

    assert row is not None
    assert row["names"] == ["fer_finger_joint1"]
    assert row["width"] == pytest.approx(0.08)


def test_windowed_rtf_does_not_cross_clock_resets() -> None:
    receive = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    simulation = np.asarray([0.0, 0.125, 0.25, 0.0, 0.125, 0.25, 0.375])

    _, rtf = _windowed_clock_rtf(receive, simulation, window_sec=0.5)

    assert rtf.tolist() == pytest.approx([0.5, 0.5])
