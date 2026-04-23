import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from rosgraph_msgs.msg import Clock
from rcl_interfaces.msg import SetParametersResult
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import numpy as np
import rclpy.time


class MultiRobotExplorer(Node):
    unreachable_frontiers = {}  # key: (robot_name, cell), value: last_close_time
    blacklisted_frontiers = set()  # key: (robot_name, cell)
    unreachable_frontiers = {}  # key: (robot_name, cell), value: (last_close_time, count)
    def __init__(self):
        super().__init__('multi_robot_explorer')

        # Parameters
        self.declare_parameter('sim_time', True)
        self.sim_time = self.get_parameter('sim_time').get_parameter_value().bool_value
        self.declare_parameter('min_unknown_cells', 15)
        self.min_unknown_cells = self.get_parameter('min_unknown_cells').get_parameter_value().integer_value

        self.add_on_set_parameters_callback(self.update_parameter_callback)

        # Time and TF
        self.latest_clock = None
        if self.sim_time:
            self.create_subscription(Clock, '/clock', self.clock_callback, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Map subscriptions
        self.create_subscription(OccupancyGrid, '/map', self.global_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/robot_1/map', self.robot1_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/robot_2/map', self.robot2_map_callback, 10)

        # Goal publishers
        self.pub_1 = self.create_publisher(PoseStamped, '/robot_1/goal_pose', 10)
        self.pub_2 = self.create_publisher(PoseStamped, '/robot_2/goal_pose', 10)

        # Marker publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)

        # Exploration timer
        self.timer = self.create_timer(2.0, self.explore)

        # Maps
        self.global_map = None
        self.local_map_1 = None
        self.local_map_2 = None

    def update_parameter_callback(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == 'min_unknown_cells' and param.type_ == rclpy.Parameter.Type.INTEGER:
                self.min_unknown_cells = param.value
                self.get_logger().info(f'Updating minimum unknown cells threshold to {self.min_unknown_cells}')
                return result
        return result

    def clock_callback(self, msg):
        self.latest_clock = msg.clock

    def global_map_callback(self, msg):
        self.global_map = msg

    def robot1_map_callback(self, msg):
        self.local_map_1 = msg

    def robot2_map_callback(self, msg):
        self.local_map_2 = msg

    def is_frontier_unreachable(self, robot_name, cell, map_msg):
        key = (robot_name, cell)
        if key in self.blacklisted_frontiers:
            return True

        now = self.get_clock().now().nanoseconds / 1e9
        dist = self.distance_to_cell(cell, map_msg, f'{robot_name}/base_link', f'{robot_name}/map')
        if dist > 1.0:
            self.unreachable_frontiers.pop(key, None)
            return False

        if key not in self.unreachable_frontiers:
            self.unreachable_frontiers[key] = now
            return False

        if now - self.unreachable_frontiers[key] > 10.0:
            self.get_logger().warn(f"Permanently blacklisting unreachable frontier {cell} for {robot_name}")
            self.blacklisted_frontiers.add(key)
            return True

        return False

    def explore(self):
        if not self.global_map or not self.local_map_1 or not self.local_map_2:
            self.get_logger().warn("Waiting for all maps...")
            return

        mask_1 = self.compute_reachability_mask(self.global_map, 'robot_1/map')
        mask_2 = self.compute_reachability_mask(self.global_map, 'robot_2/map')
        reachable_mask = np.logical_or(mask_1, mask_2)
        global_frontiers = self.find_frontiers(self.global_map, reachable_mask)
        self.publish_frontier_markers(global_frontiers, self.global_map, "global_frontiers", source_frame="world")

        self.get_logger().info(f"Global frontiers remaining: {len(global_frontiers)}")
        if not global_frontiers:
            self.get_logger().info("No frontiers left in global map. Exploration complete.")
            self.timer.cancel()
            return

        local_frontiers_1 = [f for f in self.find_frontiers(self.local_map_1) if not self.is_frontier_unreachable('robot_1', f, self.local_map_1)]
        local_frontiers_2 = [f for f in self.find_frontiers(self.local_map_2) if not self.is_frontier_unreachable('robot_2', f, self.local_map_2)]
        self.publish_frontier_markers(local_frontiers_1, self.local_map_1, "robot_1_frontiers", source_frame="robot_1/map")
        self.publish_frontier_markers(local_frontiers_2, self.local_map_2, "robot_2_frontiers", source_frame="robot_2/map")

        if local_frontiers_1:
            local_frontiers_1.sort(key=lambda cell: self.distance_to_cell(cell, self.local_map_1, 'robot_1/base_link', 'robot_1/map'))
            self.send_goal(local_frontiers_1[0], self.local_map_1, 'robot_1/map', self.pub_1)
        if local_frontiers_2:
            local_frontiers_2.sort(key=lambda cell: self.distance_to_cell(cell, self.local_map_2, 'robot_2/base_link', 'robot_2/map'))
            self.send_goal(local_frontiers_2[0], self.local_map_2, 'robot_2/map', self.pub_2)

    def distance_to_cell(self, cell, map_msg, robot_frame, source_frame):
        y, x = cell
        resolution = map_msg.info.resolution
        origin = map_msg.info.origin.position
        cx = origin.x + (x + 0.5) * resolution
        cy = origin.y + (y + 0.5) * resolution
        try:
            transform = self.tf_buffer.lookup_transform(
                source_frame, robot_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            rx = transform.transform.translation.x
            ry = transform.transform.translation.y
            return np.hypot(cx - rx, cy - ry)
        except Exception as e:
            self.get_logger().warn(f"TF transform failed for distance calculation: {e}")
            return float('inf')

    def send_goal(self, cell, map_msg, source_frame, pub):
        y, x = cell
        resolution = map_msg.info.resolution
        origin = map_msg.info.origin.position

        local_goal = PoseStamped()
        local_goal.header.frame_id = source_frame
        local_goal.header.stamp = self.latest_clock if self.sim_time and self.latest_clock else self.get_clock().now().to_msg()
        local_goal.pose.position.x = origin.x + (x + 0.5) * resolution
        local_goal.pose.position.y = origin.y + (y + 0.5) * resolution
        local_goal.pose.position.z = 0.0
        local_goal.pose.orientation.w = 1.0

        try:
            transform = self.tf_buffer.lookup_transform(
                'world', source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)
            )
            world_goal = PoseStamped()
            world_goal.pose = do_transform_pose(local_goal.pose, transform)
            world_goal.header.stamp = local_goal.header.stamp
            world_goal.header.frame_id = 'world'
            pub.publish(world_goal)

            self.get_logger().info(f"Sent goal to {pub.topic} at ({world_goal.pose.position.x:.2f}, {world_goal.pose.position.y:.2f})")
        except Exception as e:
            self.get_logger().warn(f"TF transform failed from {source_frame} to world: {e}")

    def compute_reachability_mask(self, map_msg, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                map_msg.header.frame_id, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)
            )
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f"Could not get TF for reachability mask: {e}")
            return np.ones((map_msg.info.height, map_msg.info.width), dtype=bool)

        resolution = map_msg.info.resolution
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        sx = int((tx - ox) / resolution)
        sy = int((ty - oy) / resolution)

        height = map_msg.info.height
        width = map_msg.info.width
        data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))
        reachable = np.zeros((height, width), dtype=bool)
        visited = np.zeros((height, width), dtype=bool)

        if not (0 <= sx < width and 0 <= sy < height):
            self.get_logger().warn("Robot start pose out of map bounds for reachability")
            return reachable

        queue = [(sy, sx)]
        while queue:
            y, x = queue.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            if data[y, x] != 0:
                continue
            reachable[y, x] = True
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                        queue.append((ny, nx))
        return reachable

    def find_frontiers(self, map_msg, reachable_mask=None):
        height = map_msg.info.height
        width = map_msg.info.width
        data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))
        frontiers = []

        for y in range(2, height - 2):
            for x in range(2, width - 2):
                if data[y, x] != 0:
                    continue
                if reachable_mask is not None and not reachable_mask[y, x]:
                    continue
                neighborhood = data[y-1:y+2, x-1:x+2].flatten()
                if -1 not in neighborhood:
                    continue
                unknown_area = data[y-2:y+3, x-2:x+3]
                if np.sum(unknown_area == -1) < self.min_unknown_cells:
                    continue
                if np.sum(unknown_area == 100) > 2:
                    continue
                frontiers.append((y, x))
        return frontiers

    def publish_frontier_markers(self, frontiers, map_msg, ns="frontiers", source_frame="world"):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.latest_clock if self.sim_time and self.latest_clock else self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0

        if ns == "global_frontiers":
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
        elif ns == "robot_1_frontiers":
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif ns == "robot_2_frontiers":
            marker.color.r = 0.1
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

        marker.lifetime = Duration(sec=2)
        resolution = map_msg.info.resolution
        origin = map_msg.info.origin.position

        try:
            transform = self.tf_buffer.lookup_transform(
                'world', source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed from {source_frame} to world: {e}")
            return

        for y, x in frontiers:
            local_x = origin.x + (x + 0.5) * resolution
            local_y = origin.y + (y + 0.5) * resolution
            pose = PoseStamped()
            pose.header.frame_id = source_frame
            pose.pose.position.x = local_x
            pose.pose.position.y = local_y
            pose.pose.position.z = 0.1
            pose.pose.orientation.w = 1.0
            try:
                transformed = do_transform_pose(pose.pose, transform)
                marker.points.append(Point(
                    x=transformed.position.x,
                    y=transformed.position.y,
                    z=transformed.position.z
                ))
            except Exception as e:
                self.get_logger().warn(f"Transforming marker point failed: {e}")

        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
