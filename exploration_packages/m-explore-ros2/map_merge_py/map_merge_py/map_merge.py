import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rosgraph_msgs.msg import Clock
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Header
import tf2_ros
import numpy as np
import cv2
import math
import threading
import tf_transformations
import time

class MultiRobotMapMerger(Node):
    def __init__(self):
        super().__init__('multi_robot_map_merger')

        self.declare_parameter('tf_publish_frequency', 20.0)
        self.declare_parameter('map_publish_frequency', 1.0)
        self.declare_parameter('sim_time', True)
        self.declare_parameter('visualize', True)
        self.declare_parameter('match_confidence_threshold', 65.0)

        self.publish_frequency = self.get_parameter('tf_publish_frequency').get_parameter_value().double_value
        self.map_publish_frequency = self.get_parameter('map_publish_frequency').get_parameter_value().double_value
        self.use_sim_time = self.get_parameter('sim_time').get_parameter_value().bool_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.confidence_threshold = self.get_parameter('match_confidence_threshold').get_parameter_value().double_value

        self.add_on_set_parameters_callback(self.update_parameter_callback)

        self.robot1_pos = (0.0, 0.0)
        self.robot2_pos = (0.0, 0.0)

        if self.use_sim_time:
            self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
            self.sim_time = None
            self.create_subscription(Clock, '/clock', self.clock_callback, 10)

        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)

        self.timer = self.create_timer(1.0 / self.publish_frequency, self.timer_callback)
        self.map_timer = self.create_timer(1.0 / self.map_publish_frequency, self.map_publish_callback)

        self.map1_img = None
        self.map2_img = None
        self.map1_info = None
        self.map2_info = None
        self.merged_map_img = None

        self.create_subscription(OccupancyGrid, '/robot_1/map', self.map1_callback, 10)
        self.create_subscription(OccupancyGrid, '/robot_2/map', self.map2_callback, 10)

        if self.visualize:
            self.vis_thread = threading.Thread(target=self.visualization_loop, daemon=True)
            self.vis_thread.start()

    def update_parameter_callback(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == 'match_confidence_threshold' and param.type_ == rclpy.Parameter.Type.DOUBLE:
                self.confidence_threshold = param.value
                self.get_logger().info(f'Updating minimum confidence threshold to {self.confidence_threshold}')
                return result
        # Return success, so updates are seen via get_parameter()
        return result

    def clock_callback(self, msg):
        self.sim_time = msg.clock

    def occupancy_grid_to_image(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img = np.zeros((height, width), dtype=np.uint8)
        img[data == -1] = 127
        img[data == 0] = 255
        img[data > 0] = 0
        return img

    def preprocess_image(self, img):
        return cv2.medianBlur(img, 3)

    def check_map_overlap_orb(self, img1, img2):
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0, None, None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            return 0.0, None, None, None

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 60]
        confidence = len(good_matches)
        return confidence, kp1, kp2, good_matches

    def map1_callback(self, msg):
        self.map1_img = self.occupancy_grid_to_image(msg)
        self.map1_info = msg.info

    def map2_callback(self, msg):
        self.map2_img = self.occupancy_grid_to_image(msg)
        self.map2_info = msg.info

    def timer_callback(self):
        if self.use_sim_time:
            if self.sim_time is None:
                return
            now = self.sim_time
        else:
            now = self.get_clock().now().to_msg()

        # Robot 1 TF
        tf1 = TransformStamped()
        tf1.header.stamp = now
        tf1.header.frame_id = 'world'
        tf1.child_frame_id = 'robot_1/map'
        tf1.transform.translation.x = self.robot1_pos[0]
        tf1.transform.translation.y = self.robot1_pos[1]
        tf1.transform.translation.z = 0.0
        tf1.transform.rotation.x = 0.0
        tf1.transform.rotation.y = 0.0
        tf1.transform.rotation.z = 0.0
        tf1.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(tf1)

        # Robot 2 TF with rotation
        tf2 = TransformStamped()
        tf2.header.stamp = now
        tf2.header.frame_id = 'world'
        tf2.child_frame_id = 'robot_2/map'
        tf2.transform.translation.x = self.robot2_pos[0]
        tf2.transform.translation.y = self.robot2_pos[1]
        tf2.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, getattr(self, 'robot2_yaw', 0.0))
        tf2.transform.rotation.x = q[0]
        tf2.transform.rotation.y = q[1]
        tf2.transform.rotation.z = q[2]
        tf2.transform.rotation.w = q[3]
        self.broadcaster.sendTransform(tf2)

    def map_publish_callback(self):
        if self.map1_img is None or self.map2_img is None:
            return

        img1 = self.preprocess_image(self.map1_img)
        img2 = self.preprocess_image(self.map2_img)
        confidence, kp1, kp2, good_matches = self.check_map_overlap_orb(img1, img2)

        self.get_logger().info(f"ORB good matches count: {confidence:.0f}")
        if confidence < self.confidence_threshold:
            self.get_logger().info("Using fallback layout (maps not aligned)")
        else:
            self.get_logger().info("Merging maps based on feature alignment")

        res = self.map1_info.resolution

        if confidence < self.confidence_threshold:
            gap_pixels = int(1.0 / res)
            width = max(self.map1_img.shape[1], self.map2_img.shape[1])
            height = self.map1_img.shape[0] + gap_pixels + self.map2_img.shape[0]
            canvas = np.full((height, width), 127, dtype=np.uint8)
            canvas[0:self.map1_img.shape[0], 0:self.map1_img.shape[1]] = self.map1_img
            map2_offset_y = self.map1_img.shape[0] + gap_pixels
            canvas_region = canvas[map2_offset_y:map2_offset_y + self.map2_img.shape[0], 0:self.map2_img.shape[1]]
            mask = self.map2_img != 127
            canvas_region[mask] = self.map2_img[mask]
            self.robot1_pos = (-self.map1_info.origin.position.x, -self.map1_info.origin.position.y)
            self.robot2_pos = (-self.map2_info.origin.position.x, map2_offset_y * res - self.map2_info.origin.position.y)
            self.robot2_yaw = 0.0
        else:
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if M is None:
                return
            warped = cv2.warpAffine(self.map2_img, M, (self.map1_img.shape[1], self.map1_img.shape[0]), borderValue=127)
            canvas = np.full_like(self.map1_img, 127)
            canvas[(self.map1_img == 0) | (warped == 0)] = 0
            canvas[(self.map1_img == 255) & (warped != 0) & (canvas != 0)] = 255
            canvas[(warped == 255) & (self.map1_img != 0) & (canvas != 0)] = 255
            self.robot1_pos = (-self.map1_info.origin.position.x, -self.map1_info.origin.position.y)
            # Compute correct transformed origin
            origin_px = np.array([[ -self.map2_info.origin.position.x / res,
                                     -self.map2_info.origin.position.y / res ]], dtype=np.float32)
            transformed_px = cv2.transform(origin_px[None, :, :], M)[0][0]
            self.robot2_pos = (transformed_px[0] * res, transformed_px[1] * res)

            # Store rotation for TF (in timer_callback)
            theta = math.atan2(M[1, 0], M[0, 0])
            self.robot2_yaw = theta

        self.merged_map_img = canvas

        merged_msg = OccupancyGrid()
        merged_msg.header = Header()
        merged_msg.header.stamp = self.get_clock().now().to_msg()
        merged_msg.header.frame_id = 'world'
        merged_msg.info.resolution = res
        merged_msg.info.width = canvas.shape[1]
        merged_msg.info.height = canvas.shape[0]
        merged_msg.info.origin.position.x = 0.0
        merged_msg.info.origin.position.y = 0.0
        merged_msg.info.origin.position.z = 0.0
        merged_msg.info.origin.orientation.w = 1.0

        ros_data = np.full(canvas.shape, -1, dtype=np.int8)
        ros_data[canvas == 255] = 0
        ros_data[canvas == 0] = 100
        merged_msg.data = ros_data.flatten().tolist()

        self.map_publisher.publish(merged_msg)

    def visualization_loop(self):
        while rclpy.ok():
            if self.map1_img is not None:
                cv2.imshow('Robot 1 Map', self.map1_img)
            if self.map2_img is not None:
                cv2.imshow('Robot 2 Map', self.map2_img)
            if self.merged_map_img is not None:
                cv2.imshow('Merged Map', self.merged_map_img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                rclpy.shutdown()
                break
            time.sleep(0.03)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotMapMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
