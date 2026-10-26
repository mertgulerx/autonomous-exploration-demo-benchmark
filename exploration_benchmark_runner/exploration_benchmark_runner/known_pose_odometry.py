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

"""Known-pose odometry for explorers that read their pose from an Odometry topic (TDLE).

The robot's pose in the map frame, taken from the TF ``map -> base_footprint`` (the
known pose of mapping-known-pose-v4), is published as ``nav_msgs/Odometry`` at a
fixed simulated-time rate: 10 Hz, the rate of the ground-truth odometry in TDLE's own
ROS 2 demo (docs/plans/EXTERNAL_EXPLORERS_RRT_TDLE.md, decision X2). The twist is
zero (no explorer using it reads it). Nothing is published until the transform
exists. The worker does not count this process's CPU or memory (decision X4).
"""

# The TDLE Default (config/tdle/default.yaml) names this topic as each node's odom_topic.
TOPIC = "/benchmark/known_pose_odom"
RATE_HZ = 10.0
FRAME = "map"
CHILD_FRAME = "base_footprint"


def main(args=None):
    # ROS imports stay in main, as in the robots package's nodes: the constants import without ROS.
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.time import Time
    from nav_msgs.msg import Odometry
    from tf2_ros import Buffer, TransformException, TransformListener

    class KnownPoseOdometry(Node):
        def __init__(self):
            super().__init__("known_pose_odometry")
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.output = self.create_publisher(Odometry, TOPIC, 10)
            # The node clock follows /clock with use_sim_time: a simulated-time rate.
            self.create_timer(1.0 / RATE_HZ, self.publish)

        def publish(self):
            try:
                transform = self.buffer.lookup_transform(FRAME, CHILD_FRAME, Time())
            except TransformException:
                return
            message = Odometry()
            message.header.stamp = transform.header.stamp
            message.header.frame_id = FRAME
            message.child_frame_id = CHILD_FRAME
            message.pose.pose.position.x = transform.transform.translation.x
            message.pose.pose.position.y = transform.transform.translation.y
            message.pose.pose.position.z = transform.transform.translation.z
            message.pose.pose.orientation = transform.transform.rotation
            self.output.publish(message)

    rclpy.init(args=args)
    node = KnownPoseOdometry()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
