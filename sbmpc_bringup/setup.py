from glob import glob
from setuptools import find_packages, setup


package_name = "sbmpc_bringup"


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
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/urdf", glob("urdf/*.xacro")),
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
            "cleanup_sbmpc_sim = sbmpc_bringup.cleanup_sim:main",
        ],
    },
)
