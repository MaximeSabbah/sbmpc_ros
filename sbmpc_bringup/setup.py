from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = "sbmpc_bringup"


def existing_files(pattern: str) -> list[str]:
    return sorted(path for path in glob(pattern) if Path(path).is_file())


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", existing_files("launch/*.py")),
        (f"share/{package_name}/config", existing_files("config/*.yaml")),
        (f"share/{package_name}/urdf", existing_files("urdf/*.xacro")),
        (f"share/{package_name}/mujoco", existing_files("mujoco/*.xml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sbmpc",
    maintainer_email="msabbah@laas.fr",
    description=(
        "Launch and configuration assets for bringing up SB-MPC with Franka "
        "and linear_feedback_controller."
    ),
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "sbmpc_pixi_supervisor = sbmpc_bringup.pixi_supervisor:main",
            "validate_sbmpc_sim = sbmpc_bringup.validate_sim:main",
        ],
    },
)
