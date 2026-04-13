#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from scipy.spatial.transform import Rotation
import time


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        # Subscribe to current TCP pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_raw',
            self.pose_callback,
            10
        )

        # Publish twist commands
        self.twist_pub = self.create_publisher(
            Twist,
            '/delta_twist_cmds',
            10
        )

        self.waypoints = [
            [-0.214, -0.532, 0.557, 0.707, 0.028, 0.034, 0.707],
            [0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883],
            [0.15039, -0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453],
            [0.0, 0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453],
            [-0.159, 0.501, 0.415, 0.029, 0.997, 0.045, 0.033],
            [-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008]
        ]
        self.current_wp_idx = 0
        self.pose_received = False
        self.current_pose = None
        self.waiting = False
        self.wait_start_time = None

        self.timer = self.create_timer(0.05, self.control_loop) 

        self.position_tolerance = 0.01
        self.rotation_tolerance = 0.005

        self.get_logger().info("Waypoint follower node initialized.")

    def pose_callback(self, msg):
        self.current_pose = msg
        self.pose_received = True

    def compute_twist_error(self, current_pose, target_pose):
        pos_err = np.array([
            target_pose[0] - current_pose[0],
            target_pose[1] - current_pose[1],
            target_pose[2] - current_pose[2]
        ])

        current_q = current_pose[3:]
        target_q = target_pose[3:]

        r_current = Rotation.from_quat(current_q)
        r_target = Rotation.from_quat(target_q)
        rot_error = (r_target * r_current.inv()).as_rotvec()

        Kp_pos = 1.0
        Kp_rot = 1.0

        twist = Twist()
        twist.linear.x = Kp_pos * pos_err[0]
        twist.linear.y = Kp_pos * pos_err[1]
        twist.linear.z = Kp_pos * pos_err[2]

        twist.angular.x = Kp_rot * rot_error[0]
        twist.angular.y = Kp_rot * rot_error[1]
        twist.angular.z = Kp_rot * rot_error[2]

        return twist, np.linalg.norm(pos_err), np.linalg.norm(rot_error)

    def control_loop(self):
        if not self.pose_received or self.current_wp_idx >= len(self.waypoints):
            return

        msg = self.current_pose
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        current_pose = np.concatenate((pos, quat))

        target_pose = self.waypoints[self.current_wp_idx]

        if self.waiting:
            if (self.get_clock().now().nanoseconds - self.wait_start_time) / 1e9 >= 1.0:
                self.waiting = False
                self.current_wp_idx += 1
            else:
                self.twist_pub.publish(Twist())
                return

        twist, pos_err_norm, rot_err_norm = self.compute_twist_error(current_pose, target_pose)

        if pos_err_norm < self.position_tolerance and rot_err_norm < self.rotation_tolerance:
            self.get_logger().info(f"Reached waypoint {self.current_wp_idx + 1}/{len(self.waypoints)}. Waiting 1 second...")
            if self.current_wp_idx == len(self.waypoints) - 1:
                self.get_logger().info("Task completed!")
            self.waiting = True
            self.wait_start_time = self.get_clock().now().nanoseconds
            self.twist_pub.publish(Twist())
            return

        self.twist_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
