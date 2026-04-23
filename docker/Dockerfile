FROM docker.io/osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    WORKSPACE_DIR=/opt/benchmark_ws \
    QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
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
    ros-${ROS_DISTRO}-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true
RUN rosdep update --rosdistro ${ROS_DISTRO}

WORKDIR ${WORKSPACE_DIR}
COPY . ${WORKSPACE_DIR}

RUN /bin/bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    apt-get update && \
    rosdep install --from-paths exploration_packages simulation rviz --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*"

RUN /bin/bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install"

RUN chmod +x ${WORKSPACE_DIR}/launch.sh ${WORKSPACE_DIR}/docker/entrypoint.sh

ENTRYPOINT ["/opt/benchmark_ws/docker/entrypoint.sh"]
CMD ["./launch.sh"]
