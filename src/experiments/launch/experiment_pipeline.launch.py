import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    stage = LaunchConfiguration('stage').perform(context).strip().lower()

    valid_stages = ['yolo', 'fusion', 'ackermann', 'unicycle']
    if stage not in valid_stages:
        raise RuntimeError(
            f"Valore non valido per 'stage': {stage}. "
            f"Valori ammessi: yolo, fusion, ackermann, unicycle"
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

    ackermann_launch = os.path.join(
        get_package_share_directory('ackermann_kf_tracker'),
        'launch',
        'ackermann_kf_tracker.launch.py'
    )

    unicycle_launch = os.path.join(
        get_package_share_directory('unicycle_kf_tracker'),
        'launch',
        'unicycle_kf_tracker.launch.py'
    )

    actions = []
    actions.append(LogInfo(msg=f'[experiments] Stage selezionato: {stage}'))

    # YOLO sempre attivo in tutti gli stage
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(yolo_launch)
        )
    )

    # FUSION attiva in tutti gli stage tranne yolo
    if stage in ['fusion', 'ackermann', 'unicycle']:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(fusion_launch)
            )
        )

    # Tracker Ackermann
    if stage == 'ackermann':
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(ackermann_launch)
            )
        )

    # Tracker Unicycle
    if stage == 'unicycle':
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(unicycle_launch)
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'stage',
            default_value='yolo',
            description='Valori ammessi: yolo | fusion | ackermann | unicycle'
        ),
        OpaqueFunction(function=launch_setup)
    ])

