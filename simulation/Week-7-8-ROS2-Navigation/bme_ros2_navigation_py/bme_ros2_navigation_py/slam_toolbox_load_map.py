import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from slam_toolbox.srv import DeserializePoseGraph
from geometry_msgs.msg import Pose2D
import os

class MapLoaderNode(Node):
    def __init__(self):
        super().__init__('map_loader_node')

        # Declare and get the 'map_file' parameter
        self.declare_parameter('map_file', 'bme_ros2_navigation/maps/serialized')
        map_file_param = self.get_parameter('map_file').get_parameter_value().string_value

        # Extract package name and map file path from the parameter
        package_name, map_file_path = map_file_param.split('/', 1)

        # Get the absolute path to the map file
        try:
            package_path = get_package_share_directory(package_name)
            map_file = os.path.join(package_path, map_file_path)
            self.get_logger().info(f'Using map file: {map_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to find package: {package_name}. Error: {e}')
            return

        initial_pose = Pose2D()
        initial_pose.x = 0.0
        initial_pose.y = 0.0
        initial_pose.theta = 0.0

        # Create a client for the /slam_toolbox/deserialize_map service
        self.client = self.create_client(DeserializePoseGraph, 'slam_toolbox/deserialize_map')

        # Wait for the service to become available
        self.get_logger().info('Waiting for /slam_toolbox/deserialize_map service...')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # Call the service with the map file
        # PROCESS_FIRST_NODE = 1,
        # PROCESS_NEAR_REGION = 2,
        # PROCESS_LOCALIZATION = 3
        self.request = DeserializePoseGraph.Request()
        self.request.filename = map_file
        self.request.match_type = 1
        self.request.initial_pose = initial_pose

    def send_request(self):
        self.get_logger().info(f'Calling /slam_toolbox/deserialize_map with {self.request.filename}...')
        return self.client.call_async(self.request)

def main(args=None):
    rclpy.init(args=args)
    node = MapLoaderNode()
    future = node.send_request()
    rclpy.spin_until_future_complete(node, future)

    node.get_logger().info('Map loaded successfully.')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()