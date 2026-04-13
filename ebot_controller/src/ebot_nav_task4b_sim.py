#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
import numpy as np

class WaypointController(Node):

    def __init__(self):
        super().__init__('waypoint_controller')

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.detection_status_sub = self.create_subscription(String, '/detection_status', self.detection_callback, 10)
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.imu_sub = self.create_subscription(Float32, '/orientation', self.imu_callback, 10)

        self.pose = None
        self.stop_due_to_detection = False
        self.detection_hold_end_time = None
        self.imu_yaw = None
        self.pause = False
        self.resume = False


        self.waypoints = [
            (0.43, -5.44, 1.40),
            (0.37, -1.95, 1.57),
            (0.37, 1.35, 1.57),
            (-1.66, 1.24, 3.14),  
            (-1.50, -5.60, -1.57),
            (-3.55, -5.63, 1.57),
            (-3.47, 1.31, 1.57),
            (-1.30, 1.24, 0.0),
            (-1.53, -6.61, -1.57 )
        ]

        self.current_waypoint_idx = 0

        # Control gains
        self.kp_linear = 1.0
        self.kp_angular = 1.0

        # Obstacle detection parameters
        self.obstacle_distance_threshold_front = 0.4  
        self.obstacle_distance_threshold_side = 0.4 
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
        self.imu_yaw = msg.data
        if self.imu_yaw > np.pi:
            self.imu_yaw -= 2 * np.pi

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
        valid_front = front[np.isfinite(front)]
        valid_left = left[np.isfinite(left)]
        valid_right = right[np.isfinite(right)]

        # valid_front = np.median(valid_front) if len(valid_front) > 0 else float('inf')
        # valid_left = np.median(valid_left) if len(valid_left) > 0 else float('inf')
        # valid_right = np.median(valid_right) if len(valid_right) > 0 else float('inf')

        self.obstacle_detected_front = np.any((valid_front < self.obstacle_distance_threshold_front) & (valid_front > 0))
        self.obstacle_detected_left = np.any((valid_left < self.obstacle_distance_threshold_side) & (valid_left > 0))
        self.obstacle_detected_right = np.any((valid_right < self.obstacle_distance_threshold_side) & (valid_right > 0))

        self.left_clearance = np.nanmean(valid_left) if len(valid_left) > 0 else float('inf')
        self.right_clearance = np.nanmean(valid_right) if len(valid_right) > 0 else float('inf')

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
            self.get_logger().info("Obstacle detected in front")
            twist.linear.x = 0.0
            twist.angular.z = 0.75 if self.left_clearance > self.right_clearance else -0.75
            self.cmd_pub.publish(twist)
            return

        elif self.obstacle_detected_left:
            self.get_logger().info("Obstacle detected on left")
            twist.linear.x = 0.5
            twist.angular.z = -0.4  # turn right
            self.cmd_pub.publish(twist)
            return

        elif self.obstacle_detected_right:
            self.get_logger().info("Obstacle detected on right")
            twist.linear.x = 0.5
            twist.angular.z = 0.4  # turn left
            self.cmd_pub.publish(twist)
            return

        # Navigation to waypoint
        goal_x, goal_y, goal_yaw = self.waypoints[self.current_waypoint_idx]
        dx = goal_x - self.pose[0]
        dy = goal_y - self.pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        yaw = self.imu_yaw if self.imu_yaw is not None else self.pose[2]
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
                self.get_logger().info(f"Position reached at waypoint {self.current_waypoint_idx}, fixing yaw.")
        else:
            if abs(final_yaw_error) > 0.05:
                twist.linear.x = 0.0
                twist.angular.z = np.clip(self.kp_angular * final_yaw_error, -1.0, 1.0)
            else:
                self.get_logger().info(f"Waypoint {self.current_waypoint_idx} reached.")
                self.get_logger().info(f"{self.pose}")

                if self.current_waypoint_idx == 1:
                    msg = String()
                    msg.data = f"DOCK_STATION,{self.pose[0]:.2f},{self.pose[1]:.2f},0"
                    self.detection_pub.publish(msg)

                self.fixing_yaw = False
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    self.goal_reached = True
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info("Waypoint Navigation Completed.")
                else:
                    self.get_logger().info(f"Moving to waypoint {self.current_waypoint_idx}.")

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
