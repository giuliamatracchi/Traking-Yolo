from setuptools import setup
from glob import glob

package_name = 'ackermann_kf_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/ackermann_kf_tracker.yaml']),
        ('share/' + package_name + '/launch', ['launch/ackermann_kf_tracker.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Ackermann EKF tracker using YOLO track_id + depth',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ackermann_kf_tracker = ackermann_kf_tracker.ackermann_kf_tracker_node:main',
        ],
    },
)

