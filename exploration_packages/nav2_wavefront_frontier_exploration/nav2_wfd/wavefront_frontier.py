#!/usr/bin/env python3
"""Wavefront Frontier Exploration node for ROS 2 Jazzy + Nav2.

This node consumes the SLAM occupancy grid (`/map`), extracts frontier centroids
from unknown-space boundaries, and sends goals to Nav2 `navigate_to_pose`.
"""

from __future__ import annotations

from collections import deque
import math
import sys
from typing import Deque, Iterable, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


UNKNOWN_COST = -1
FREE_COST = 0


class OccupancyGrid2D:
    """Small helper wrapper around nav_msgs/OccupancyGrid."""

    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid

    @property
    def width(self) -> int:
        return self.grid.info.width

    @property
    def height(self) -> int:
        return self.grid.info.height

    @property
    def resolution(self) -> float:
        return self.grid.info.resolution

    def in_bounds(self, mx: int, my: int) -> bool:
        return 0 <= mx < self.width and 0 <= my < self.height

    def index(self, mx: int, my: int) -> int:
        return my * self.width + mx

    def get_cost(self, mx: int, my: int) -> int:
        return self.grid.data[self.index(mx, my)]

    def map_to_world(self, mx: int, my: int) -> tuple[float, float]:
        wx = self.grid.info.origin.position.x + (mx + 0.5) * self.resolution
        wy = self.grid.info.origin.position.y + (my + 0.5) * self.resolution
        return wx, wy

    def world_to_map(self, wx: float, wy: float) -> tuple[int, int]:
        ox = self.grid.info.origin.position.x
        oy = self.grid.info.origin.position.y
        if wx < ox or wy < oy:
            raise ValueError("World coordinates out of map bounds")

        mx = int((wx - ox) / self.resolution)
        my = int((wy - oy) / self.resolution)
        if not self.in_bounds(mx, my):
            raise ValueError("World coordinates out of map bounds")
        return mx, my


def neighbors8(mx: int, my: int, grid: OccupancyGrid2D) -> Iterable[tuple[int, int]]:
    for nx in range(mx - 1, mx + 2):
        for ny in range(my - 1, my + 2):
            if nx == mx and ny == my:
                continue
            if grid.in_bounds(nx, ny):
                yield nx, ny


def is_frontier_cell(
    mx: int,
    my: int,
    grid: OccupancyGrid2D,
    occupied_threshold: int,
) -> bool:
    if grid.get_cost(mx, my) != UNKNOWN_COST:
        return False

    has_free_neighbor = False
    for nx, ny in neighbors8(mx, my, grid):
        cost = grid.get_cost(nx, ny)
        if cost >= occupied_threshold:
            return False
        if cost == FREE_COST:
            has_free_neighbor = True

    return has_free_neighbor


def find_nearest_free_cell(
    start: tuple[int, int],
    grid: OccupancyGrid2D,
) -> tuple[int, int]:
    if grid.get_cost(start[0], start[1]) == FREE_COST:
        return start

    q: Deque[tuple[int, int]] = deque([start])
    visited = {start}
    while q:
        cx, cy = q.popleft()
        for nx, ny in neighbors8(cx, cy, grid):
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if grid.get_cost(nx, ny) == FREE_COST:
                return nx, ny
            q.append((nx, ny))

    return start


def collect_frontier_cluster(
    seed: tuple[int, int],
    grid: OccupancyGrid2D,
    occupied_threshold: int,
    visited_frontier: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cluster: list[tuple[int, int]] = []
    q: Deque[tuple[int, int]] = deque([seed])
    visited_frontier.add(seed)

    while q:
        cx, cy = q.popleft()
        if not is_frontier_cell(cx, cy, grid, occupied_threshold):
            continue

        cluster.append((cx, cy))
        for nx, ny in neighbors8(cx, cy, grid):
            if (nx, ny) in visited_frontier:
                continue
            if not is_frontier_cell(nx, ny, grid, occupied_threshold):
                continue
            visited_frontier.add((nx, ny))
            q.append((nx, ny))

    return cluster


def frontier_clusters_from_pose(
    robot_pose: PoseStamped,
    grid: OccupancyGrid2D,
    occupied_threshold: int,
) -> list[list[tuple[int, int]]]:
    start = grid.world_to_map(robot_pose.pose.position.x, robot_pose.pose.position.y)
    start_free = find_nearest_free_cell(start, grid)

    free_q: Deque[tuple[int, int]] = deque([start_free])
    visited_free = {start_free}
    visited_frontier: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []

    while free_q:
        fx, fy = free_q.popleft()

        for nx, ny in neighbors8(fx, fy, grid):
            ncost = grid.get_cost(nx, ny)
            if ncost == FREE_COST and (nx, ny) not in visited_free:
                visited_free.add((nx, ny))
                free_q.append((nx, ny))

            if (nx, ny) in visited_frontier:
                continue

            if is_frontier_cell(nx, ny, grid, occupied_threshold):
                cluster = collect_frontier_cluster(
                    (nx, ny),
                    grid,
                    occupied_threshold,
                    visited_frontier,
                )
                if cluster:
                    clusters.append(cluster)

    return clusters


class WavefrontFrontierExplorer(Node):
    def __init__(self) -> None:
        super().__init__("nav2_wavefront_frontier_explorer")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("navigate_to_pose_action", "navigate_to_pose")
        self.declare_parameter("local_costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("local_costmap_obstacle_threshold", 100)
        self.declare_parameter("planner_frequency_hz", 1.0)
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("min_frontier_size", 5)
        self.declare_parameter("min_goal_distance_m", 0.35)
        self.declare_parameter("frontier_marker_topic", "/explore/frontiers")
        self.declare_parameter("initial_pose_topic", "/explore/path_tracker/initial_pose")

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        self.navigate_to_pose_action = str(self.get_parameter("navigate_to_pose_action").value)
        self.local_costmap_topic = str(self.get_parameter("local_costmap_topic").value)
        self.local_costmap_obstacle_threshold = int(
            self.get_parameter("local_costmap_obstacle_threshold").value
        )
        self.planner_frequency_hz = float(self.get_parameter("planner_frequency_hz").value)
        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self.min_frontier_size = int(self.get_parameter("min_frontier_size").value)
        self.min_goal_distance_m = float(self.get_parameter("min_goal_distance_m").value)
        self.frontier_marker_topic = str(self.get_parameter("frontier_marker_topic").value)
        self.initial_pose_topic = str(self.get_parameter("initial_pose_topic").value)

        if self.planner_frequency_hz <= 0.0:
            raise RuntimeError("planner_frequency_hz must be > 0")
        if not (0 <= self.occupied_threshold <= 100):
            raise RuntimeError("occupied_threshold must be in [0, 100]")
        if self.min_frontier_size < 1:
            raise RuntimeError("min_frontier_size must be >= 1")
        if not (0 <= self.local_costmap_obstacle_threshold <= 100):
            raise RuntimeError("local_costmap_obstacle_threshold must be in [0, 100]")

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        local_costmap_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            local_costmap_qos,
        )
        self.frontier_pub = self.create_publisher(MarkerArray, self.frontier_marker_topic, 10)
        self.initial_pose_pub = self.create_publisher(PoseStamped, self.initial_pose_topic, latched_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_action = ActionClient(self, NavigateToPose, self.navigate_to_pose_action)

        self.map_grid: Optional[OccupancyGrid2D] = None
        self.local_costmap_grid: Optional[OccupancyGrid2D] = None
        self.initial_pose_published = False
        self.goal_in_progress = False
        self.goal_handle = None

        self._throttle_cache: dict[str, int] = {}

        self.timer = self.create_timer(
            1.0 / self.planner_frequency_hz,
            self.plan_once,
        )

        self.get_logger().info(
            "Wavefront explorer ready "
            f"(map='{self.map_topic}', action='{self.navigate_to_pose_action}', "
            f"frame='{self.global_frame}', base='{self.robot_base_frame}')"
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.map_grid = OccupancyGrid2D(msg)

    def local_costmap_callback(self, msg: OccupancyGrid) -> None:
        self.local_costmap_grid = OccupancyGrid2D(msg)

    def get_robot_pose(self) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                Time(),
                timeout=Duration(seconds=0.3),
            )
        except TransformException as exc:
            self._throttled_log(
                key="wait_tf",
                interval_ns=5_000_000_000,
                message=f"Waiting for TF: {exc}",
                level="warn",
            )
            return None

        pose = PoseStamped()
        pose.header = transform.header
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def plan_once(self) -> None:
        if self.map_grid is None or self.goal_in_progress:
            return

        if not self.nav_action.wait_for_server(timeout_sec=0.1):
            self._throttled_log(
                key="wait_action_server",
                interval_ns=2_000_000_000,
                message=f"Waiting for '{self.navigate_to_pose_action}' action server...",
            )
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        if not self.initial_pose_published:
            self.initial_pose_pub.publish(robot_pose)
            self.initial_pose_published = True

        try:
            clusters = frontier_clusters_from_pose(
                robot_pose,
                self.map_grid,
                self.occupied_threshold,
            )
        except ValueError as exc:
            self._throttled_log(
                key="pose_out_of_map",
                interval_ns=2_000_000_000,
                message=f"Robot pose is outside map bounds: {exc}",
                level="warn",
            )
            return

        centroids = self.to_world_centroids(clusters)
        self.publish_frontier_markers(centroids)

        if not centroids:
            self._throttled_log(
                key="no_frontier",
                interval_ns=3_000_000_000,
                message="No new frontiers found",
            )
            return

        goal = self.select_goal(centroids, robot_pose)
        if goal is None:
            self._throttled_log(
                key="all_frontiers_too_close",
                interval_ns=3_000_000_000,
                message="Frontiers exist, but all are below minimum goal distance",
            )
            return

        if self.is_goal_blocked_in_local_costmap(goal):
            self._throttled_log(
                key="goal_blocked_local_costmap",
                interval_ns=2_000_000_000,
                message="Selected frontier is blocked in local costmap; canceling goal dispatch",
                level="warn",
            )
            return

        self.send_goal(goal)

    def to_world_centroids(self, clusters: list[list[tuple[int, int]]]) -> list[tuple[float, float]]:
        assert self.map_grid is not None

        centroids: list[tuple[float, float]] = []
        for cluster in clusters:
            if len(cluster) < self.min_frontier_size:
                continue

            sum_x = 0.0
            sum_y = 0.0
            for mx, my in cluster:
                wx, wy = self.map_grid.map_to_world(mx, my)
                sum_x += wx
                sum_y += wy

            n = float(len(cluster))
            centroids.append((sum_x / n, sum_y / n))

        return centroids

    def select_goal(
        self,
        centroids: list[tuple[float, float]],
        robot_pose: PoseStamped,
    ) -> Optional[tuple[float, float]]:
        best: Optional[tuple[float, float]] = None
        best_distance = float("inf")

        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y

        for wx, wy in centroids:
            distance = math.hypot(wx - rx, wy - ry)
            if distance < self.min_goal_distance_m:
                continue
            if distance < best_distance:
                best_distance = distance
                best = (wx, wy)

        if best is not None:
            return best

        # Fallback: if every frontier is near the robot, still pick the nearest.
        best_distance = float("inf")
        for wx, wy in centroids:
            distance = math.hypot(wx - rx, wy - ry)
            if distance < best_distance:
                best_distance = distance
                best = (wx, wy)
        return best

    def send_goal(self, goal_xy: tuple[float, float]) -> None:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.global_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_xy[0]
        goal_msg.pose.pose.position.y = goal_xy[1]
        goal_msg.pose.pose.orientation.w = 1.0

        self.goal_in_progress = True
        self.get_logger().info(f"Sending goal: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
        future = self.nav_action.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def transform_xy_between_frames(
        self,
        x: float,
        y: float,
        source_frame: str,
        target_frame: str,
    ) -> Optional[tuple[float, float]]:
        if source_frame == target_frame:
            return (x, y)

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self._throttled_log(
                key="wait_goal_local_costmap_tf",
                interval_ns=2_000_000_000,
                message=(
                    f"Waiting for transform '{source_frame}' -> '{target_frame}' "
                    f"to validate local costmap: {exc}"
                ),
                level="warn",
            )
            return None

        q = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        x_out = math.cos(yaw) * x - math.sin(yaw) * y + tx
        y_out = math.sin(yaw) * x + math.cos(yaw) * y + ty
        return x_out, y_out

    def is_goal_blocked_in_local_costmap(self, goal_xy: tuple[float, float]) -> bool:
        if self.local_costmap_grid is None:
            return False

        costmap_frame = self.local_costmap_grid.grid.header.frame_id or self.global_frame
        transformed_goal = self.transform_xy_between_frames(
            goal_xy[0],
            goal_xy[1],
            self.global_frame,
            costmap_frame,
        )
        if transformed_goal is None:
            return False

        try:
            mx, my = self.local_costmap_grid.world_to_map(transformed_goal[0], transformed_goal[1])
        except ValueError:
            return False

        cost = self.local_costmap_grid.get_cost(mx, my)
        return cost >= self.local_costmap_obstacle_threshold

    def goal_response_callback(self, future) -> None:
        try:
            self.goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.goal_in_progress = False
            self.get_logger().error(f"Goal send failed: {exc}")
            return

        if self.goal_handle is None or not self.goal_handle.accepted:
            self.goal_in_progress = False
            self.get_logger().warn("Goal rejected by action server")
            return

        self.get_logger().info("Goal accepted")
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future) -> None:
        self.goal_in_progress = False
        self.goal_handle = None

        try:
            wrapped_result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to get goal result: {exc}")
            return

        status = wrapped_result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal completed successfully")
            return

        self.get_logger().warn(f"Goal failed/canceled, status={status}")

    def publish_frontier_markers(self, centroids: list[tuple[float, float]]) -> None:
        marker_array = MarkerArray()

        clear = Marker()
        clear.header.frame_id = self.global_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for idx, (wx, wy) in enumerate(centroids):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "wavefront_frontiers"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            marker.color.r = 0.0
            marker.color.g = 0.9
            marker.color.b = 0.2
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        self.frontier_pub.publish(marker_array)

    def _throttled_log(
        self,
        key: str,
        interval_ns: int,
        message: str,
        level: str = "info",
    ) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self._throttle_cache.get(key, 0)
        if now_ns - last_ns < interval_ns:
            return
        self._throttle_cache[key] = now_ns

        if level == "warn":
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WavefrontFrontierExplorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
