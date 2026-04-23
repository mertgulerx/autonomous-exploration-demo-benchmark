import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import quaternion_from_euler
import time

class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')

        # Declare and get parameters for the initial position and orientation
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', -4.0)
        self.declare_parameter('yaw', 0.0)

        x = self.get_parameter('x').get_parameter_value().double_value
        y = self.get_parameter('y').get_parameter_value().double_value
        yaw = self.get_parameter('yaw').get_parameter_value().double_value

        # Hardcoded covariance matrix
        covariance_matrix = [
            3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 3.0
        ]

        # Create a publisher for the /initialpose topic
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Create the PoseWithCovarianceStamped message
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.frame_id = 'map'
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.pose.pose.position.x = x
        initial_pose_msg.pose.pose.position.y = y
        initial_pose_msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        quaternion = quaternion_from_euler(0, 0, yaw)
        initial_pose_msg.pose.pose.orientation.x = quaternion[0]
        initial_pose_msg.pose.pose.orientation.y = quaternion[1]
        initial_pose_msg.pose.pose.orientation.z = quaternion[2]
        initial_pose_msg.pose.pose.orientation.w = quaternion[3]

        # Set the covariance matrix
        initial_pose_msg.pose.covariance = covariance_matrix

        # Publish the message
        self.get_logger().info('Publishing initial pose...')
        self.publisher.publish(initial_pose_msg)
        self.get_logger().info('Initial pose published.')

def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
    # Give some time to publish before shutdown
    time.sleep(2)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()