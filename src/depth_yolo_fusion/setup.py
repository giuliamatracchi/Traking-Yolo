


from setuptools import setup

package_name = 'depth_yolo_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/depth_yolo_fusion.yaml']),
        ('share/' + package_name + '/launch', ['launch/depth_yolo_fusion.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Fusion node for DepthCamera and YOLO detections',
    license='MIT',
    entry_points={
        'console_scripts': [
            'depth_yolo_fusion = depth_yolo_fusion.depth_yolo_fusion_node:main',
        ],
    },
)

