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

from setuptools import setup
from glob import glob

setup(
    name="exploration_benchmark_runner",
    version="0.1.0",
    packages=["exploration_benchmark_runner"],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/exploration_benchmark_runner"],
        ),
        ("share/exploration_benchmark_runner", ["package.xml", "LICENSE"]),
        (
            "share/exploration_benchmark_runner/config",
            [
                "config/profiles.yaml",
                "config/comparison.yaml",
                "config/parameters.yaml",
                "config/worlds.yaml",
            ],
        ),
        (
            "share/exploration_benchmark_runner/config/coverage",
            glob("config/coverage/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mert Güler",
    maintainer_email="support.mertgulerx@gmail.com",
    description="Isolated exploration benchmark supervision",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "coordinator = exploration_benchmark_runner.coordinator:main",
            "worker = exploration_benchmark_runner.worker:main",
            "benchmark = exploration_benchmark_runner.cli:main",
            "benchmark_reference = exploration_benchmark_runner.reference_builder:main",
        ]
    },
)
