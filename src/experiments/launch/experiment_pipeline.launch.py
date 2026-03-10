import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    stage = LaunchConfiguration('stage').perform(context).strip().lower()

    valid_stages = ['yolo', 'fusion', 'tracker']
    if stage not in valid_stages:
        raise RuntimeError(
            f"Valore non valido per 'stage': {stage}. "
            f"Valori ammessi: yolo, fusion, tracker"
        )

    yolo_launch = os.path.join(
        get_package_share_directory('yolo_detector'),
        'launch',
        'yolo_detector.launch.py'
    )

    fusion_launch = os.path.join(
        get_package_share_directory('depth_yolo_fusion'),
        'launch',
        'depth_yolo_fusion.launch.py'
    )

    tracker_launch = os.path.join(
        get_package_share_directory('ackermann_kf_tracker'),
        'launch',
        'ackermann_kf_tracker.launch.py'
    )

    actions = []

    actions.append(LogInfo(msg=f'[experiments] Stage selezionato: {stage}'))

    # stage=yolo -> solo YOLO
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(yolo_launch)
        )
    )

    # stage=fusion -> YOLO + depth fusion
    if stage in ['fusion', 'tracker']:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(fusion_launch)
            )
        )

    # stage=tracker -> YOLO + depth fusion + tracker
    if stage == 'tracker':
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(tracker_launch)
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'stage',
            default_value='yolo',
            description='Valori ammessi: yolo | fusion | tracker'
        ),
        OpaqueFunction(function=launch_setup)
    ])

