#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory("trajectory_metrics")
    config_file = os.path.join(package_dir, "config", "trajectory_metrics.yaml")

    return LaunchDescription([
        Node(
            package="trajectory_metrics",
            executable="trajectory_metrics_node",
            name="trajectory_metrics_node",
            output="screen",
            parameters=[config_file],
        )
    ])

