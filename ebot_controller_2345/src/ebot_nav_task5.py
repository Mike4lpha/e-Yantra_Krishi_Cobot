#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String
from std_msgs.msg import Bool
from tf_transformations import euler_from_quaternion
import signal
import numpy as np
import math

class WaypointController(Node):

    def __init__(self):
        super().__init__('waypoint_controller')

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.detection_status_sub = self.create_subscription(String, '/detection_status', self.detection_callback, 10)
        self.imu_sub = self.create_subscription(Float32, '/orientation', self.imu_callback, 10)
        self.item_sub = self.create_subscription(Bool, '/fertilizer_delivered', self.fertilizer_callback, 10)

        self.dock_reached_pub = self.create_publisher(Bool, '/dock_reached', 10)
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.stop_pub = self.create_publisher(Bool, '/stop_detection', 10)
        

        self.pose = None
        self.stop_due_to_detection = False
        self.detection_hold_end_time = None
        self.imu_yaw = None
        self.pause = False
        self.resume = False
        self.shutdown_requested = False
        self.waiting_for_item = False
        self.received_item = False

        self.waypoints = [
            (-0.024, -1.652, -1.57),
            (-0.005, -1.652, 0.00), ## starting of lane 1
            (2.3158, -1.666, 0.00), ## dock station (2.0158)
            (4.947, -1.760, 0.00), ## 4.90 end of lane 1
            (4.947, -1.760, 1.57),
            (4.70, 0.1, 1.57),
            (4.70, 0.1, 3.14), ## start of lane 2
            (0.50, -0.04, 3.14), ## end of lane 2
            (0.58, 1.70, 0.00), ## start of lane 3
            (4.70, 1.80, 0.00), ## end of lane 3
            (4.70, 0.0, -1.57),
            (0.0, 0.0, 3.14)
        ]

        self.current_waypoint_idx = 0

        # Control gains
        self.kp_linear = 1.0
        self.kp_angular = 1.0

        # Obstacle detection parameters
        self.obstacle_distance_threshold_front = 0.0
        self.obstacle_distance_threshold_side = 0.0
        self.obstacle_detected_front = False
        self.obstacle_detected_left = False
        self.obstacle_detected_right = False
        self.left_clearance = float('inf')
        self.right_clearance = float('inf')

        self.fixing_yaw = False

        self.timer = self.create_timer(0.1, self.control_loop)

        self.goal_reached = False

        self.get_logger().info("Waypoint Navigation started.")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])
        self.pose = [pos.x, pos.y, yaw]
        # self.get_logger().info(f"{self.pose[0]}, {self.pose[1]}, {self.pose[2]}")

    def imu_callback(self, msg: Float32):
        orientation = msg.data
        self.imu_yaw = math.atan2(math.sin(orientation), math.cos(orientation))

    def fertilizer_callback(self, msg):
        if msg.data:
            self.get_logger().info("Received confirmation...fertilzer loaded.")
            self.received_item = True
            self.waiting_for_item = False
            self.post_delivery_end_time = self.get_clock().now().seconds_nanoseconds()[0] + 2.0

    def scan_sectors(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        num_ranges = len(ranges)

        def angle_to_index(angle_rad):
            return int((angle_rad - angle_min) / angle_increment)

        right_start = angle_to_index(np.deg2rad(-100))
        right_end   = angle_to_index(np.deg2rad(-45))
        front_start = angle_to_index(np.deg2rad(-45))
        front_end   = angle_to_index(np.deg2rad(45))
        left_start  = angle_to_index(np.deg2rad(45))
        left_end    = angle_to_index(np.deg2rad(100))

        right_ranges = ranges[max(0, right_start):min(num_ranges, right_end)]
        front_ranges = ranges[max(0, front_start):min(num_ranges, front_end)]
        left_ranges  = ranges[max(0, left_start):min(num_ranges, left_end)]

        return left_ranges, front_ranges, right_ranges

    def scan_callback(self, msg: LaserScan):
        left, front, right = self.scan_sectors(msg)

        # Filter invalid values
        self.valid_front = front[np.isfinite(front)]
        self.valid_left = left[np.isfinite(left)]
        self.valid_right = right[np.isfinite(right)]

        self.obstacle_detected_front = np.any((self.valid_front < self.obstacle_distance_threshold_front) & (self.valid_front > 0))
        self.obstacle_detected_left = np.any((self.valid_left < self.obstacle_distance_threshold_side) & (self.valid_left > 0))
        self.obstacle_detected_right = np.any((self.valid_right < self.obstacle_distance_threshold_side) & (self.valid_right > 0))

        self.left_clearance = np.nanmean(self.valid_left) if len(self.valid_left) > 0 else float('inf')
        self.right_clearance = np.nanmean(self.valid_right) if len(self.valid_right) > 0 else float('inf')

    def detection_callback(self, msg: String):
        self.get_logger().info(f"Detected: {msg.data}")
        self.stop_due_to_detection = True
        now = self.get_clock().now().nanoseconds * 1e-9
        self.detection_hold_end_time = now + 2.0
        self.publish_stop()

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


    def control_loop(self):

        # if self.imu_yaw is None:
        #     self.publish_stop()
        #     self.get_logger().warn("Waiting for IMU yaw...")
        #     self.publish_stop()
        #     return
        if self.waiting_for_item and not self.received_item:
            self.publish_stop()
            return

        # If a detection pause is active
        if self.stop_due_to_detection:
            current_time = self.get_clock().now().nanoseconds * 1e-9

            if not self.pause:
                self.get_logger().info("Paused")
                self.pause = True
                self.resume = False

            if current_time < self.detection_hold_end_time:
                self.publish_stop()
                return
            else:
                if not self.resume:
                    self.get_logger().info("Resuming")
                    self.resume = True

                self.stop_due_to_detection = False
                self.pause = False
                return

        if self.pose is None or self.goal_reached:
            self.publish_stop()
            return

        twist = Twist()

        # Obstacle avoidance
        if self.obstacle_detected_front:
            self.get_logger().info(f"Obstacle detected in front")
            twist.linear.x = 0.0
            twist.angular.z = 0.75 if self.left_clearance > self.right_clearance else -0.75
            self.cmd_pub.publish(twist)
            return

        elif self.obstacle_detected_left:
            self.get_logger().info(f"Obstacle detected on left")
            twist.linear.x = 0.5
            twist.angular.z = -0.4  # turn right
            self.cmd_pub.publish(twist)
            return

        elif self.obstacle_detected_right:
            self.get_logger().info(f"Obstacle detected on right")
            twist.linear.x = 0.5
            twist.angular.z = 0.4  # turn left
            self.cmd_pub.publish(twist)
            return

        # Navigation to waypoint
        goal_x, goal_y, goal_yaw = self.waypoints[self.current_waypoint_idx]
        dx = goal_x - self.pose[0]
        dy = goal_y - self.pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        yaw = self.pose[2]
        # yaw = self.imu_yaw
        angle_to_goal = np.arctan2(dy, dx)

        # Fix yaw error
        yaw_error = np.arctan2(np.sin(angle_to_goal - yaw), np.cos(angle_to_goal - yaw))
        final_yaw_error = np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw))

        if not self.fixing_yaw:
            if distance > 0.1:
                if abs(yaw_error) > np.deg2rad(10):
                    twist.linear.x = 0.0
                    twist.angular.z = np.clip(self.kp_angular * yaw_error, -1.0, 1.0)
                else:
                    twist.linear.x = np.clip(self.kp_linear * distance, 0.0, 0.5)
                    twist.angular.z = np.clip(self.kp_angular * yaw_error, -1.0, 1.0)
            else:
                self.fixing_yaw = True
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(f"Position reached at waypoint {self.current_waypoint_idx + 1}, fixing yaw.")
        else:
            if abs(final_yaw_error) > 0.05:
                twist.linear.x = 0.0
                twist.angular.z = np.clip(self.kp_angular * final_yaw_error, -1.0, 1.0)
            else:
                self.get_logger().info(f"Waypoint {self.current_waypoint_idx + 1} reached.")
                self.get_logger().info(f"{self.pose}")

                if self.current_waypoint_idx == 2:
                    msg = String()
                    msg.data = f"DOCK_STATION,{self.pose[0]:.2f},{self.pose[1]:.2f},0"
                    self.detection_pub.publish(msg)

                    dock_msg = Bool()
                    dock_msg.data = True
                    self.dock_reached_pub.publish(dock_msg)

                    self.get_logger().info("Dock reached...Waiting for fertilizer")

                    self.waiting_for_item = True
                    self.received_item = False

                if self.current_waypoint_idx == 9:
                    msg = Bool()
                    msg.data = True
                    self.stop_pub.publish(msg)
                    self.get_logger().info("Shape detection DISABLED")

                self.fixing_yaw = False
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    self.goal_reached = True
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info("Waypoint Navigation Completed.")
                else:
                    self.get_logger().info(f"Moving to waypoint {self.current_waypoint_idx + 1}.")

        self.cmd_pub.publish(twist)

    def request_shutdown(self):
        if self.shutdown_requested:
            return

        self.shutdown_requested = True
        self.get_logger().warn("Shutdown requested. Stopping robot.")
        self.publish_stop()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    def sigint_handler(sig, frame):
        node.request_shutdown()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)

    rclpy.spin(node)

if __name__ == '__main__':
    main()
