from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data= {'yolo_detector': ['*.yaml']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install model under a dedicated models/ folder in package share
        (os.path.join('share', package_name, 'models'), ['yolov8n.pt']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOU',
    maintainer_email='you@example.com',
    description='YOLOv8 detector node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector = yolo_detector.yolo_detector_node:main',
        ],
    },
)