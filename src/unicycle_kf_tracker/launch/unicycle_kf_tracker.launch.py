#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('unicycle_kf_tracker')
    config_file = os.path.join(package_dir, 'config', 'unicycle_kf_tracker.yaml')

    return LaunchDescription([
        Node(
            package='unicycle_kf_tracker',
            executable='unicycle_kf_tracker',
            name='unicycle_kf_tracker',
            output='screen',
            parameters=[config_file],
        )
    ])

