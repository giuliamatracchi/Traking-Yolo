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
            executable='yolo_detector',  # deve corrispondere alla entry point in setup.py
            name='YoloDetector',
            parameters=[{
                # Parametri che compongono il topic di input:
                'env_topic': 'environment',
                'sensor_type': 'RGBCamera',
                'specific_name': 'CameraSDT',
                'topic_raw': 'image_raw',

                # (OPZIONALE) Se vuoi forzare un topic assoluto diverso, decommenta:
                # 'input_topic': '/environment/RGBCamera/CameraAlt/image_raw',

                # Output & modello
                'output_image_topic': 'yolo/annotated_image',
                'output_detections_topic': 'yolo/detections',
                'model_path': model_path,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'class_filter': [0,2]  # es. [0,2] per person & car
            }],
            # remappings=[  # di norma non serve più, ma resta qui per completezza
            #     ('/camera/image', '/environment/RGBCamera/CameraSDT/image_raw'),
            # ]
        )
    ])

