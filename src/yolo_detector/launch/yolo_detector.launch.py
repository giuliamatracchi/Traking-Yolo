from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Modello installato in share/yolo_detector/models/yolov8n.pt (vedi setup.py)
    model_path = os.path.join(
        get_package_share_directory('yolo_detector'), 'models', 'yolov8n.pt'
    )

    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='yolo_detector',
            name='YoloDetector',
            parameters=[{
                'output_image_topic': 'yolo/annotated_image',
                'output_detections_topic': 'yolo/detections',
                'model_path': model_path,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'tracker_cfg': '',
            }],
        )
    ])

