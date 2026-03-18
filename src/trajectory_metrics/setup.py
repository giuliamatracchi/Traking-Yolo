from setuptools import setup
from glob import glob
import os

package_name = "trajectory_metrics"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="isarlab",
    maintainer_email="isarlab@example.com",
    description="ROS2 node for trajectory metrics against GPS ground truth",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "trajectory_metrics_node = trajectory_metrics.trajectory_metrics_node:main",
        ],
    },
)

