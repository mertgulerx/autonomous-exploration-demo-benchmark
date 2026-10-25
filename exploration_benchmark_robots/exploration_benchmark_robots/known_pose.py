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

"""Known-pose mapping condition: ground-truth odom->base_footprint TF.

The Gazebo PosePublisher gives the robot model's world pose. The model frame is
base_footprint for every robot (``model_frame`` refuses anything else), so this
node turns the planar model pose into the TF ``odom -> base_footprint``, with
``odom`` the world frame at the commanded spawn pose (x, y, yaw), so SLAM
Toolbox's ``map -> odom`` starts as the identity. No odometry topic is
published; the robot's own odometry topics stay the measurement source.

The node also checks that it is the only source of base_footprint's parent
(``ConflictCheck``). A foreign transform (an EKF, a drive plugin or a controller
left publishing, or a static transform) is counted and reported on the latched
status topic, which the worker checks.
"""

from collections import OrderedDict
import json
import math
import xml.etree.ElementTree as ET

# Condition-defining constants of MAPPING_CONDITION (with mapping_scan.FILL_OFFSET_M
# and THRESHOLD_MARGIN_M): changing one of them requires a new MAPPING_CONDITION id.
# Recorded per run in robot_configuration.comparison_profile.mapping_parameters.
# v1: stock SLAM Toolbox 2.8.5; v2: a Cartographer trial, reverted (tuning runs
# only); v3: SLAM Toolbox 2.8.5 with the grid-extent patch, no-return beams free to
# the range threshold (tuning runs only); v4: the same patch with no-return beams
# free for NO_RETURN_FREE_LENGTH_M only (docker/patches, plan I10 §8).
MAPPING_CONDITION = "mapping-known-pose-v4"
# Free length of a no-return beam (Cartographer's default missing_data_ray_length): a
# real lidar also returns nothing from glass, mirrors and dark or grazed surfaces.
# Compiled into the patch as kBenchmarkNoReturnFreeLength; keep the two equal.
NO_RETURN_FREE_LENGTH_M = 5.0
SLAM_TOOLBOX_VERSION = "2.8.5"
SLAM_TOOLBOX_PATCH = "slam_toolbox-2.8.5-grid-extent.patch"
SLAM_TOOLBOX_PATCH_SHA256 = (
    "381910851b8d625e982b870937318e489282f8463b7db93fc42b83a4940c6b8b"
)
# Written next to the patched build's share files by the Dockerfiles.
SLAM_TOOLBOX_PATCH_MARKER = "share/slam_toolbox/benchmark_grid_extent.sha256"
# 100 Hz: an exact 5:1 multiple of the recorded 20 Hz ground_truth stream on both
# physics steps in use (0.01 s and 0.001 s); 50 Hz would not decimate to 20 Hz.
GROUND_TRUTH_RATE_HZ = 100
# The suite's ground_truth stream keeps the first pose of every 50 ms of sim time.
GROUND_TRUTH_RECORD_PERIOD_NS = 50_000_000
GROUND_TRUTH_TOPIC = "/benchmark/ground_truth/pose"
MAPPING_STATUS_TOPIC = "/benchmark/mapping/status"
BASE_FRAME = "base_footprint"
# Own transforms kept for the conflict check (10 s of ground truth at 100 Hz).
OWN_HISTORY = 1000
# Status cadence of both mapping nodes: on change, else every 1 s of message stamp time.
STATUS_PERIOD_NS = 1_000_000_000


def ground_truth_settings():
    """The PosePublisher settings (model world pose only)."""
    return dict(
        topic=GROUND_TRUTH_TOPIC,
        publish_model_pose=True,
        publish_nested_model_pose=False,
        publish_link_pose=False,
        publish_sensor_pose=False,
        publish_collision_pose=False,
        publish_visual_pose=False,
        use_pose_vector_msg=True,
        static_publisher=False,
        update_frequency=GROUND_TRUTH_RATE_HZ,
    )


def add_ground_truth_publisher(root):
    """Add the PosePublisher to a URDF robot or an SDF model; returns its settings."""
    model = ET.SubElement(root, "gazebo") if root.tag == "robot" else root.find("model")
    if model is None:
        raise ValueError("Ground-truth pose requires a URDF robot or an SDF model")
    plugin = ET.SubElement(
        model,
        "plugin",
        filename="gz-sim-pose-publisher-system",
        name="gz::sim::systems::PosePublisher",
    )
    settings = ground_truth_settings()
    for name, value in settings.items():
        ET.SubElement(plugin, name).text = (
            str(value).lower() if isinstance(value, bool) else str(value)
        )
    return settings


def ground_truth_bridge():
    return dict(
        ros_topic_name=GROUND_TRUTH_TOPIC,
        gz_topic_name=GROUND_TRUTH_TOPIC,
        ros_type_name="tf2_msgs/msg/TFMessage",
        gz_type_name="gz.msgs.Pose_V",
        direction="GZ_TO_ROS",
    )


def model_frame(spawn_root, description_root, base_frame=BASE_FRAME):
    """The robot_state_publisher link at the Gazebo model origin (``base_frame``).

    A URDF spawn becomes a model whose frame is the URDF root link. An SDF spawn
    must contain a link named like the URDF root without its own pose (so at the
    model frame). A root other than ``base_frame`` would need a model-to-base
    offset, which no adapter has: it is refused.
    """
    links = [link.get("name") for link in description_root.findall("link")]
    children = {
        joint.find("child").get("link") for joint in description_root.findall("joint")
    }
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"Robot description needs exactly one root link: {roots}")
    root = roots[0]
    if root != base_frame:
        raise ValueError(
            f"Robot description root {root!r} is not {base_frame!r}; "
            "the ground-truth offset needs a dedicated adapter"
        )
    if spawn_root.tag == "robot":
        return root
    model = spawn_root.find("model")
    link = None if model is None else model.find(f"link[@name='{root}']")
    pose = None if link is None else link.find("pose")
    if link is None or (
        pose is not None
        and (
            pose.get("relative_to")
            or any(float(v) for v in (pose.text or "").split())
        )
    ):
        raise ValueError(
            f"SDF model has no link {root!r} at its model frame; "
            "the ground-truth offset needs a dedicated adapter"
        )
    return root


def yaw(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def compose(a, b):
    x, y, angle = a
    c, s = math.cos(angle), math.sin(angle)
    return x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], wrap(angle + b[2])


def inverse(a):
    x, y, angle = a
    c, s = math.cos(angle), math.sin(angle)
    return -c * x - s * y, s * x - c * y, wrap(-angle)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def odom_origin(spawn_pose):
    """World pose (x, y, yaw) of ``odom``: base_footprint at the commanded spawn pose.

    Under the known-pose condition ``map -> odom`` is the identity, so this is also
    the exact ``world_from_map`` of the coverage alignment (worker.prepare_coverage).
    """
    x, y, _, angle = spawn_pose
    return float(x), float(y), wrap(float(angle))


class ConflictCheck:
    """Whether a transform naming base_footprint as its child came from this node.

    Every own ``odom -> base_footprint`` is remembered (``own``) before it is sent.
    Foreign: any static transform to base_footprint, any other parent of
    base_footprint, and, once own history exists, an ``odom -> base_footprint``
    whose stamp is not in the history (stamp 0 and stamps older than the history
    included) or whose value differs from the one sent with that stamp. Before the
    first own transform an ``odom -> base_footprint`` cannot be judged. A second
    publisher sending exactly the own pose with the own stamp is indistinguishable.
    """

    def __init__(self, odom_frame="odom", base_frame=BASE_FRAME, history=OWN_HISTORY):
        self.odom_frame, self.base_frame, self.history = odom_frame, base_frame, history
        self.sent = OrderedDict()

    def own(self, stamp_ns, value):
        self.sent[stamp_ns] = value
        while len(self.sent) > self.history:
            self.sent.popitem(last=False)

    def foreign(self, parent, child, stamp_ns, value, static=False):
        if child.lstrip("/") != self.base_frame:
            return False
        if static or parent.lstrip("/") != self.odom_frame:
            return True
        if not self.sent:
            return False
        return self.sent.get(stamp_ns) != value


def run_node(factory, args=None):
    """Spin a mapping node until SIGINT or SIGTERM, then let it report its final status.

    rclpy's own signal handlers shut the context down before a node could publish,
    so they are replaced by a flag that ends a spin loop with a short timeout.
    """
    import signal

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    requested = []
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda received, _frame: requested.append(received))
    node = factory()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not requested and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    finally:
        try:
            node.final_report()
        except Exception as error:  # an invalid context must not hide the shutdown
            print(f"{node.get_name()}: final status not published: {error}", flush=True)
        executor.shutdown(timeout_sec=1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    from geometry_msgs.msg import TransformStamped
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
    from tf2_ros import TransformBroadcaster

    latched = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    # tf2's static listener profile: late joiners get every static transform.
    static = QoSProfile(
        depth=100,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class KnownPose(Node):
        def __init__(self):
            super().__init__("known_pose_tf")
            self.model = self.declare_parameter("model", "").value
            self.model_frame = self.declare_parameter("model_frame", "").value
            self.base_frame = self.declare_parameter("base_frame", BASE_FRAME).value
            self.odom_frame = self.declare_parameter("odom_frame", "odom").value
            self.spawn_pose = list(
                self.declare_parameter("spawn_pose", [0.0, 0.0, 0.0, 0.0]).value
            )
            if not self.model or not self.model_frame or len(self.spawn_pose) != 4:
                raise ValueError("known_pose_tf needs model, model_frame, spawn_pose")
            if self.model_frame != self.base_frame:
                raise ValueError(
                    f"known_pose_tf needs the model frame at {self.base_frame}, "
                    f"not {self.model_frame}"
                )
            self.world_odom = odom_origin(self.spawn_pose)
            self.conflicts = ConflictCheck(self.odom_frame, self.base_frame)
            self.published = 0
            self.foreign = 0
            self.foreign_sample = None
            self.detail = "waiting for the ground-truth pose"
            self.last_status = None
            self.last_status_ns = None
            self.status = self.create_publisher(String, MAPPING_STATUS_TOPIC, latched)
            self.broadcaster = TransformBroadcaster(self)
            self.create_subscription(
                TFMessage, GROUND_TRUTH_TOPIC, self.receive, qos_profile_sensor_data
            )
            self.create_subscription(
                TFMessage, "/tf", self.check, qos_profile_sensor_data
            )
            self.create_subscription(
                TFMessage,
                "/tf_static",
                lambda message: self.check(message, static=True),
                static,
            )
            self.get_logger().info(f"world->{self.odom_frame} {self.world_odom}")
            self.report(0, force=True)

        def receive(self, message):
            for item in message.transforms:
                if item.child_frame_id.split("::")[-1] != self.model:
                    continue
                stamp = item.header.stamp
                stamp_ns = stamp.sec * 10**9 + stamp.nanosec
                t, r = item.transform.translation, item.transform.rotation
                values = (t.x, t.y, r.x, r.y, r.z, r.w)
                if not all(math.isfinite(value) for value in values):
                    self.detail = "non-finite ground-truth pose skipped"
                    self.report(stamp_ns)
                    return
                base = (t.x, t.y, yaw(r.x, r.y, r.z, r.w))
                x, y, angle = compose(inverse(self.world_odom), base)
                out = TransformStamped()
                out.header.stamp = stamp
                out.header.frame_id = self.odom_frame
                out.child_frame_id = self.base_frame
                out.transform.translation.x = x
                out.transform.translation.y = y
                out.transform.rotation.z = math.sin(angle / 2)
                out.transform.rotation.w = math.cos(angle / 2)
                self.conflicts.own(stamp_ns, (x, y, out.transform.rotation.z))
                self.broadcaster.sendTransform(out)
                self.published += 1
                if self.foreign == 0:
                    self.detail = "publishing"
                self.report(stamp_ns)
                return

        def check(self, message, static=False):
            for item in message.transforms:
                stamp = item.header.stamp
                stamp_ns = stamp.sec * 10**9 + stamp.nanosec
                t = item.transform
                if not self.conflicts.foreign(
                    item.header.frame_id,
                    item.child_frame_id,
                    stamp_ns,
                    (t.translation.x, t.translation.y, t.rotation.z),
                    static,
                ):
                    continue
                self.foreign += 1
                first = self.foreign_sample is None
                if first:
                    self.foreign_sample = dict(
                        parent=item.header.frame_id,
                        child=item.child_frame_id,
                        static=static,
                        stamp_ns=stamp_ns,
                        x=t.translation.x,
                        y=t.translation.y,
                    )
                    self.get_logger().error(
                        f"Second publisher of {self.base_frame}'s parent "
                        f"(TF conflict): {self.foreign_sample}"
                    )
                self.detail = f"TF conflict: another {self.base_frame} parent publisher"
                self.report(stamp_ns, force=first)

        def snapshot(self):
            return dict(
                node="known_pose_tf",
                ok=self.published > 0 and self.foreign == 0,
                detail=self.detail,
                published=self.published,
                foreign=self.foreign,
                foreign_sample=self.foreign_sample,
                world_odom=self.world_odom,
                model_frame=self.model_frame,
            )

        def report(self, stamp_ns, force=False):
            status = self.snapshot()
            state = (status["ok"], self.foreign > 0, self.detail)
            if (
                not force
                and state == self.last_status
                and self.last_status_ns is not None
                and 0 <= stamp_ns - self.last_status_ns < STATUS_PERIOD_NS
            ):
                return
            self.last_status, self.last_status_ns = state, stamp_ns
            self.status.publish(String(data=json.dumps(status)))

        def final_report(self):
            status = dict(self.snapshot(), final=True)
            self.get_logger().info("Final status: " + json.dumps(status))
            self.status.publish(String(data=json.dumps(status)))

    run_node(KnownPose, args)


if __name__ == "__main__":
    main()
