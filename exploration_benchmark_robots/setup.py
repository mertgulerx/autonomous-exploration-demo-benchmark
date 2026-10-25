# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from glob import glob
from setuptools import setup

setup(
    name="exploration_benchmark_robots",
    version="0.1.0",
    packages=["exploration_benchmark_robots"],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/exploration_benchmark_robots"],
        ),
        ("share/exploration_benchmark_robots", ["package.xml", "LICENSE"]),
        (
            "share/exploration_benchmark_robots/config",
            glob("config/*.yaml") + glob("config/*.json"),
        ),
        ("share/exploration_benchmark_robots/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mert Güler",
    maintainer_email="support.mertgulerx@gmail.com",
    description="Robot profiles and isolated simulation adapters",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "command_bridge = exploration_benchmark_robots.command_bridge:main",
            "known_pose_tf = exploration_benchmark_robots.known_pose:main",
            "mapping_scan_filter = exploration_benchmark_robots.mapping_scan:main",
            "robot_dependencies = exploration_benchmark_robots.dependencies:main",
            "robot_geometry = exploration_benchmark_robots.geometry:main",
        ]
    },
)
