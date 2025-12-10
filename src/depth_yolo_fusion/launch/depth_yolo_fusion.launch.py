from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('depth_yolo_fusion'),
        'config',
        'depth_yolo_fusion.yaml'
    )

    node = Node(
        package='depth_yolo_fusion',
        executable='depth_yolo_fusion',
        name='depth_yolo_fusion_node',
        output='screen',
        parameters=[config],
    )

    return LaunchDescription([node])
