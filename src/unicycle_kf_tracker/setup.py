from setuptools import setup
from glob import glob
import os

package_name = 'unicycle_kf_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='isarlab',
    maintainer_email='isarlab@example.com',
    description='Unicycle EKF tracker node for fused YOLO + depth detections',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'unicycle_kf_tracker = unicycle_kf_tracker.unicycle_kf_tracker_node:main',
        ],
    },
)

