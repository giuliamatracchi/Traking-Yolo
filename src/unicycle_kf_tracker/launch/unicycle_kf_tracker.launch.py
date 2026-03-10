from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('unicycle_kf_tracker')
    cfg = os.path.join(pkg_share, 'config', 'unicycle_kf_tracker.yaml')

    return LaunchDescription([
        Node(
            package='unicycle_kf_tracker',
            executable='unicycle_kf_tracker',
            name='unicycle_kf_tracker',
            output='screen',
            parameters=[cfg],
        )
    ])

