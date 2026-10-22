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

FROM docker.io/osrf/ros@sha256:ce169d5e9c87bcbddf01685366184671f8d6d6329893456cc14ae1ab6c062eb2
LABEL org.benchmark.multi_robot="1"

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    WORKSPACE_DIR=/opt/benchmark_ws \
    BENCHMARK_ROBOT_DEPENDENCIES=/opt/vendor \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    QT_X11_NO_MITSHM=1

# Package archives frozen at the binary lock's date (exploration_benchmark_robots/config/binary_lock.yaml): the
# live Ubuntu and ROS archives keep only their newest versions, so every apt step installs from the Ubuntu and
# ROS snapshot archives and a rebuild reproduces the locked closure. The ROS snapshot key is fetched by its full
# fingerprint and checked. Refreshing the lock means new dates here and a new lock (a changed environment).
ARG UBUNTU_SNAPSHOT=20261011T000000Z
ARG ROS_SNAPSHOT=2026-10-05
ARG ROS_SNAPSHOT_KEY=4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA
RUN set -eu; \
    python3 -c "import sys, urllib.request; sys.stdout.buffer.write(urllib.request.urlopen('https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x${ROS_SNAPSHOT_KEY}', timeout=60).read())" > /tmp/ros-snapshot.asc; \
    test "$(gpg --batch --quiet --show-keys --with-colons /tmp/ros-snapshot.asc | awk -F: '$1 == "fpr" {print $10; exit}')" = "${ROS_SNAPSHOT_KEY}"; \
    gpg --batch --quiet --dearmor -o /usr/share/keyrings/ros-snapshots-archive-keyring.gpg /tmp/ros-snapshot.asc; \
    rm /tmp/ros-snapshot.asc; \
    printf 'Types: deb\nURIs: http://snapshots.ros.org/%s/%s/ubuntu\nSuites: noble\nComponents: main\nSigned-By: /usr/share/keyrings/ros-snapshots-archive-keyring.gpg\n' \
        "${ROS_DISTRO}" "${ROS_SNAPSHOT}" > /etc/apt/sources.list.d/ros2.sources; \
    sed -i "s#http://archive.ubuntu.com/ubuntu/\?#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#; s#http://security.ubuntu.com/ubuntu/\?#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" \
        /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-psutil \
    python3-numpy \
    xauth \
    x11-apps \
    mesa-utils \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxkbcommon0 \
    libwayland-client0 \
    wayland-protocols \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-slam-toolbox \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-topic-tools \
    ros-${ROS_DISTRO}-interactive-marker-twist-server \
    ros-${ROS_DISTRO}-xacro \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true
RUN rosdep update --rosdistro ${ROS_DISTRO}

WORKDIR ${WORKSPACE_DIR}
# Source and binary lock changes rebuild cached vendor layers.
COPY exploration_benchmark_robots/exploration_benchmark_robots/__init__.py exploration_benchmark_robots/exploration_benchmark_robots/dependencies.py ${WORKSPACE_DIR}/exploration_benchmark_robots/exploration_benchmark_robots/
COPY exploration_benchmark_robots/config/source_lock.yaml exploration_benchmark_robots/config/binary_lock.yaml ${WORKSPACE_DIR}/exploration_benchmark_robots/config/
COPY docker/build_robot_dependencies.sh ${WORKSPACE_DIR}/docker/

RUN /bin/bash docker/build_robot_dependencies.sh
COPY . ${WORKSPACE_DIR}

RUN /bin/bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    apt-get update && \
    rosdep install --from-paths exploration_packages simulation rviz exploration_benchmark_interfaces exploration_benchmark_robots exploration_benchmark_runner --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*"

RUN /bin/bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --symlink-install --executor sequential --cmake-args -DBUILD_TESTING=OFF"

# References use immutable Fuel versions; prepare the exact geometry before
# preflight. This starts no simulator and prevents downloads during a benchmark.
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source install/setup.bash && ros2 run exploration_benchmark_runner benchmark_reference fetch --world warehouse \
    --project-root ${WORKSPACE_DIR}"

RUN chmod +x ${WORKSPACE_DIR}/launch.sh ${WORKSPACE_DIR}/docker/entrypoint.sh

ENTRYPOINT ["/opt/benchmark_ws/docker/entrypoint.sh"]
CMD ["./launch.sh"]
