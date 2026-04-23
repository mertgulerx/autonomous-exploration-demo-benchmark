# nav2_wfd (ROS 2 Jazzy)

Wavefront Frontier Detection explorer for ROS 2 Jazzy + Nav2.

## What it does

- Subscribes to `OccupancyGrid` map (`/map`)
- Finds frontier regions (unknown/free-space boundaries)
- Publishes frontier markers on `/explore/frontiers`
- Sends goals to Nav2 `navigate_to_pose`
- Publishes the initial pose to `/explore/path_tracker/initial_pose`

## Build

From your workspace root:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build --packages-select nav2_wfd
source install/setup.bash
```

## Run

1. Start your robot simulation/stack with SLAM and Nav2 (`/map`, TF, `navigate_to_pose` must exist).
2. Run explorer:

```bash
ros2 run nav2_wfd explore --ros-args -p use_sim_time:=true
```

## Useful parameters

- `map_topic` (default: `/map`)
- `global_frame` (default: `map`)
- `robot_base_frame` (default: `base_footprint`)
- `navigate_to_pose_action` (default: `navigate_to_pose`)
- `planner_frequency_hz` (default: `1.0`)
- `occupied_threshold` (default: `50`)
- `min_frontier_size` (default: `5`)
- `min_goal_distance_m` (default: `0.35`)
- `frontier_marker_topic` (default: `/explore/frontiers`)
- `initial_pose_topic` (default: `/explore/path_tracker/initial_pose`)
