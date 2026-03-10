from setuptools import setup

package_name = 'unicycle_kf_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/unicycle_kf_tracker.yaml']),
        ('share/' + package_name + '/launch', ['launch/unicycle_kf_tracker.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='isarlab',
    maintainer_email='matracchigiulia@gmail.com',
    description='Unicycle EKF tracker using YOLO track_id + depth',
    license='MIT',
    entry_points={
        'console_scripts': [
            'unicycle_kf_tracker = unicycle_kf_tracker.unicycle_kf_tracker_node:main',
        ],
    },
)

