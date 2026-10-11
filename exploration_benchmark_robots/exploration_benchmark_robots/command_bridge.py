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

"""Convert the benchmark Twist contract to measured-time native commands."""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class CommandBridge(Node):
    def __init__(self):
        super().__init__("benchmark_native_command_bridge")
        output = self.declare_parameter("output", "/native/cmd_vel").value
        self.limits = [
            self.declare_parameter(name, 0.0).value
            for name in ("max_linear_speed", "max_lateral_speed", "max_angular_speed")
        ]
        self.publisher = self.create_publisher(TwistStamped, output, 10)
        self.create_subscription(Twist, "/cmd_vel", self.received, 10)

    def received(self, velocity):
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_footprint"
        command.twist = velocity
        # A native mecanum controller may not enforce its documented envelope.
        # These immutable platform limits also apply to non-navigation commands.
        values = (velocity.linear.x, velocity.linear.y, velocity.angular.z)
        if not all(math.isfinite(value) for value in values):
            values = (0.0, 0.0, 0.0)
        command.twist.linear.x, command.twist.linear.y, command.twist.angular.z = (
            max(-limit, min(limit, value)) for value, limit in zip(values, self.limits)
        )
        command.twist.linear.z = command.twist.angular.x = command.twist.angular.y = 0.0
        self.publisher.publish(command)


def main():
    rclpy.init()
    node = CommandBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
