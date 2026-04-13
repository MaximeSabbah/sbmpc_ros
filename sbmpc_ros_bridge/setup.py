from setuptools import find_packages, setup


package_name = "sbmpc_ros_bridge"


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
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="sbmpc",
    maintainer_email="msabbah@laas.fr",
    description=(
        "ROS 2 bridge adapters between linear_feedback_controller messages "
        "and the sbmpc planner API."
    ),
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "sbmpc_lfc_bridge_node = sbmpc_ros_bridge.lfc_bridge_node:main",
        ],
    },
)
