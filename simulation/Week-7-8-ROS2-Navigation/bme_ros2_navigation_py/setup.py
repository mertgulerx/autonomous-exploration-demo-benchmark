from setuptools import find_packages, setup

package_name = 'bme_ros2_navigation_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='David Dudas',
    maintainer_email='david.dudas@outlook.com',
    description='Python nodes for slam, localization and navigation with Gazebo Harmonic and ROS Jazzy for BME MOGI ROS2 course',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_initialpose = bme_ros2_navigation_py.send_initialpose:main',
            'slam_toolbox_load_map = bme_ros2_navigation_py.slam_toolbox_load_map:main',
            'follow_waypoints = bme_ros2_navigation_py.follow_waypoints:main',
        ],
    },
)
