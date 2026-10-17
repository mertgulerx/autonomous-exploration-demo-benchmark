#!/usr/bin/env python3
"""
Copyright 2026 Mert Güler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty as EmptySrv
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener


# Path tracker node keeps per-explorer traversed paths in separate topics while sharing
# the same TF-driven pose sampling loop. The separation makes RViz comparisons deterministic.
class FrontierPathTracker(Node):
    def __init__(self) -> None:
        super().__init__("frontier_path_tracker")

        # Parameters are intentionally explicit so the same tracker can be reused with
        # different robot frames, topic layouts, and package naming conventions.
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("path_topic", "/explore/traversed_path")
        self.declare_parameter(
            "package_topics",
            [
                "frontier_exploration_ros2_mrtsp:/explore/traversed_path",
                "frontier_exploration_ros2_nearest:/explore/traversed_path/frontier_exploration_ros2_nearest",
                "m_explore_ros2:/explore/traversed_path/m_explore_ros2",
                "nav2_wavefront_frontier_exploration:/explore/traversed_path/nav2_wavefront_frontier_exploration",
                "roadmap_explorer:/explore/traversed_path/roadmap_explorer",
            ],
        )
        self.declare_parameter("active_package_topic", "/explore/path_tracker/active_package")
        self.declare_parameter("initial_pose_topic", "/explore/path_tracker/initial_pose")
        self.declare_parameter("default_package", "frontier_exploration_ros2_mrtsp")
        self.declare_parameter("reset_topic", "/explore/reset_traveled_path")
        self.declare_parameter("update_rate_hz", 5.0)
        self.declare_parameter("min_translation_delta_m", 0.05)

        # Read once at startup; runtime reconfiguration is out of scope for this utility.
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.package_topics_raw = list(self.get_parameter("package_topics").value)
        self.active_package_topic = str(self.get_parameter("active_package_topic").value)
        self.initial_pose_topic = str(self.get_parameter("initial_pose_topic").value)
        self.default_package = str(self.get_parameter("default_package").value).strip()
        self.reset_topic = str(self.get_parameter("reset_topic").value)
        self.update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        self.min_translation_delta_m = float(
            self.get_parameter("min_translation_delta_m").value
        )

        # Guardrails prevent accidental misconfiguration from creating tight loops or
        # invalid movement thresholds that would make all pose samples pass/fail.
        if self.update_rate_hz <= 0.0:
            raise RuntimeError("update_rate_hz must be greater than 0.0")
        if self.min_translation_delta_m < 0.0:
            raise RuntimeError("min_translation_delta_m must be non-negative")

        # TF listener is the single source of robot pose so all packages share identical
        # sampled positions and differ only in how the active package is switched.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Latched path topics let RViz/new subscribers immediately receive the last path.
        # Reliable delivery is preferred because the stream is low-rate but semantically important.
        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_qos = path_qos
        self.package_topics = self._parse_package_topics(self.package_topics_raw)
        if self.default_package == "":
            self.default_package = "frontier_exploration_ros2_mrtsp"
        if self.default_package not in self.package_topics:
            self.package_topics[self.default_package] = self.path_topic

        # One publisher + cached Path message per package keeps append/publish operations
        # simple and avoids reconstructing the full path on every timer tick.
        self.path_publishers: dict[str, rclpy.publisher.Publisher] = {}
        self.path_msgs: dict[str, Path] = {}
        for package_name, topic_name in self.package_topics.items():
            self.path_publishers[package_name] = self.create_publisher(
                Path, topic_name, self.path_qos
            )
            path_msg = Path()
            path_msg.header.frame_id = self.global_frame
            self.path_msgs[package_name] = path_msg

        # Active package is switched by control panel so only one path grows at a time.
        self.active_package = self.default_package
        self.initial_pose_publisher = self.create_publisher(
            PoseStamped, self.initial_pose_topic, self.path_qos
        )
        # Reset is exposed as both topic and service for compatibility with different callers.
        self.active_package_subscriber = self.create_subscription(
            String, self.active_package_topic, self._on_active_package, 10
        )
        self.reset_subscriber = self.create_subscription(
            EmptyMsg, self.reset_topic, self._on_reset_topic, 10
        )
        self.reset_service = self.create_service(
            EmptySrv, "~/reset_path", self._on_reset_service
        )

        self.last_pose: PoseStamped | None = None
        self.initial_pose: PoseStamped | None = None
        self.last_lookup_warning_ns = 0

        # Fixed-rate sampling keeps path density stable across packages and runs.
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._record_pose)

        self.get_logger().info(
            "Frontier path tracker enabled: "
            f"frame='{self.global_frame}', "
            f"base='{self.robot_base_frame}', "
            f"default_package='{self.default_package}', "
            f"active_package_topic='{self.active_package_topic}', "
            f"initial_pose_topic='{self.initial_pose_topic}', "
            f"reset_topic='{self.reset_topic}', "
            f"rate={self.update_rate_hz:.2f} Hz"
        )

    def _parse_package_topics(self, entries: list[str]) -> dict[str, str]:
        # Input format is "package_name:topic_name". Invalid entries are ignored so
        # partially broken configs still produce a usable tracker.
        package_topics: dict[str, str] = {}
        for raw_entry in entries:
            entry = str(raw_entry).strip()
            if ":" not in entry:
                continue
            package_name, topic_name = entry.split(":", 1)
            package_name = package_name.strip()
            topic_name = topic_name.strip()
            if package_name == "" or topic_name == "":
                continue
            package_topics[package_name] = topic_name

        # Ensure at least one writable channel exists; callers can still switch dynamically later.
        if not package_topics:
            package_topics["frontier_exploration_ros2_mrtsp"] = self.path_topic
        return package_topics

    def _sanitize_package_name(self, package_name: str) -> str:
        # Topic suffixes are constrained to a conservative character set to avoid malformed topics.
        return "".join(
            [char if (char.isalnum() or char in "_-") else "_" for char in package_name]
        )

    def _ensure_package(self, package_name: str) -> str:
        # Unknown package names are created lazily to support ad-hoc experiments without restart.
        name = package_name.strip()
        if name == "":
            return self.active_package
        if name not in self.path_publishers:
            safe_name = self._sanitize_package_name(name)
            topic_name = f"/explore/traversed_path/{safe_name}"
            self.path_publishers[name] = self.create_publisher(Path, topic_name, self.path_qos)
            path_msg = Path()
            path_msg.header.frame_id = self.global_frame
            self.path_msgs[name] = path_msg
            self.get_logger().info(
                f"Created dynamic path publisher for package '{name}' on topic '{topic_name}'."
            )
        return name

    def _on_active_package(self, msg: String) -> None:
        # Active package updates are idempotent and safe even if the package is first seen now.
        package_name = self._ensure_package(msg.data)
        self.active_package = package_name
        self.get_logger().info(f"Active path package changed to '{self.active_package}'.")

    def _reset_path(self) -> None:
        # Reset clears all package paths together so comparison runs restart from identical state.
        self.last_pose = None
        stamp = self.get_clock().now().to_msg()
        for package_name, publisher in self.path_publishers.items():
            path_msg = Path()
            path_msg.header.frame_id = self.global_frame
            path_msg.header.stamp = stamp
            self.path_msgs[package_name] = path_msg
            publisher.publish(path_msg)
        self.active_package = self.default_package
        self.get_logger().info("Traveled path reset requested for all packages.")

    def _on_reset_topic(self, _: EmptyMsg) -> None:
        # Topic-based reset path is intentionally fire-and-forget.
        self._reset_path()

    def _on_reset_service(self, request: EmptySrv.Request, response: EmptySrv.Response) -> EmptySrv.Response:
        # Service variant mirrors topic behavior but provides RPC-style integration option.
        del request
        self._reset_path()
        return response

    def _record_pose(self) -> None:
        # Sample current base pose in global frame. Missing TF is expected during startup and
        # is handled by throttled warnings to avoid log spam.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                Time(),
            )
        except TransformException as exc:
            self._warn_lookup_failure(str(exc))
            return

        # Convert transform into PoseStamped so it can be appended directly to nav_msgs/Path.
        pose = PoseStamped()
        pose.header = transform.header
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation

        # Distance gate avoids dense near-duplicate points when robot is stationary or jittering.
        if not self._should_append_pose(pose):
            return

        # Initial pose is latched once so panel "return to start" has a stable reference.
        if self.initial_pose is None:
            self.initial_pose = pose
            self.initial_pose_publisher.publish(self.initial_pose)

        # Append only to currently active package path and publish updated path snapshot.
        self.last_pose = pose
        active_package = self._ensure_package(self.active_package)
        active_path = self.path_msgs[active_package]
        active_path.header.stamp = pose.header.stamp
        active_path.poses.append(pose)
        self.path_publishers[active_package].publish(active_path)

    def _should_append_pose(self, pose: PoseStamped) -> bool:
        # First valid pose always seeds the path.
        if self.last_pose is None:
            return True

        # Euclidean translation threshold keeps path size bounded by meaningful movement.
        dx = pose.pose.position.x - self.last_pose.pose.position.x
        dy = pose.pose.position.y - self.last_pose.pose.position.y
        dz = pose.pose.position.z - self.last_pose.pose.position.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return distance >= self.min_translation_delta_m

    def _warn_lookup_failure(self, message: str) -> None:
        # Warning throttle prevents flooding logs when TF is temporarily unavailable.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_lookup_warning_ns < 5_000_000_000:
            return
        self.last_lookup_warning_ns = now_ns
        self.get_logger().warn(
            "Waiting for TF transform "
            f"'{self.global_frame}' -> '{self.robot_base_frame}': {message}"
        )


def main(args=None) -> None:
    # Standard rclpy lifecycle with graceful shutdown on Ctrl+C or external executor stop.
    rclpy.init(args=args)
    node = FrontierPathTracker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
