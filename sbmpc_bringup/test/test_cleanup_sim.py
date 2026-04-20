from __future__ import annotations

from sbmpc_bringup.cleanup_sim import (
    ProcessInfo,
    find_sbmpc_sim_processes,
    is_sbmpc_sim_process,
    is_sbmpc_sim_launch_process,
    parse_ps_output,
)


def test_parse_ps_output_reads_pid_ppid_and_command() -> None:
    processes = parse_ps_output(
        " 123 1 ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py\n"
        " not-a-pid 1 ignored\n"
    )

    assert processes == (
        ProcessInfo(
            pid=123,
            ppid=1,
            command="ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py",
        ),
    )


def test_is_sbmpc_sim_process_matches_only_managed_processes() -> None:
    assert is_sbmpc_sim_process(
        "ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py use_rviz:=false"
    )
    assert is_sbmpc_sim_process("/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge")
    assert is_sbmpc_sim_process("python -m sbmpc_ros_bridge.lfc_bridge_node")
    assert not is_sbmpc_sim_process("python unrelated_script.py")
    assert is_sbmpc_sim_launch_process(
        "ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py"
    )
    assert not is_sbmpc_sim_launch_process("gz sim empty.sdf -r -s")


def test_find_sbmpc_sim_processes_excludes_current_cleanup_process() -> None:
    processes = (
        ProcessInfo(pid=10, ppid=1, command="python -m sbmpc_ros_bridge.lfc_bridge_node"),
        ProcessInfo(pid=11, ppid=10, command="python unrelated_script.py"),
    )

    assert find_sbmpc_sim_processes(processes, exclude_pids={10}) == ()
