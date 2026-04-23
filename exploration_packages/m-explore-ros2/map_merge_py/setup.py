from setuptools import find_packages, setup

package_name = 'map_merge_py'

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
    description='Python package multirobot_map_merge',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_merge = map_merge_py.map_merge:main',
            'explore_map = map_merge_py.explore_map:main',
        ],
    },
)
