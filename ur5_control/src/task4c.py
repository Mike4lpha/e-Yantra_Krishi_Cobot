#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from scipy.spatial.transform import Rotation
import tf2_ros
from tf2_ros import TransformException
from std_msgs.msg import Bool
from linkattacher_msgs.srv import AttachLink, DetachLink

TEAM_ID = "2345"
MAX_FRUIT_ID = 3


class UR5_Manipulation(Node):
    def __init__(self):
        super().__init__('ur5_manipulation_node')

        # ---------------- ROS ----------------
        self.pose_sub = self.create_subscription(
            PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)

        self.twist_pub = self.create_publisher(
            Twist, '/delta_twist_cmds', 10)

        self.dock_reached_sub = self.create_subscription(
            Bool, '/dock_reached', self.dock_reached_callback, 10)

        self.delivery_pub = self.create_publisher(
            Bool, '/fertilizer_delivered', 10)
        self.unloaded_pub = self.create_publisher(
            Bool, '/fertilizer_unloaded', 10)

        # ---------------- SERVICES ----------------
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        self.wait_for_services()

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- WAYPOINTS ----------------
        self.waypoints = [
            [-0.21661, -0.53397, 0.62345, 0.707, 0.028, 0.034, 0.707],
            [0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883],
            [0.15039, -0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453],
            [0.816, 0.11041, 0.30932, 0.71253, 0.70134, 0.016744, 0.012449],
            [0.69968, -0.0052498, 0.43932, -0.70134, 0.71253, 0.012449, -0.016744]
        ]

        self.waypoints2 = [
            # [0.15039, -0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453],
            [0.78825, 0.035833, 0.31702, 0.75759, -0.65162, -0.031134, -0.02213],
            [0.816, 0.11041, 0.30932, 0.71253, 0.70134, 0.016744, 0.012449],
            [0.15039, -0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453],
            [0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883],
            [-0.11377, -0.22327, 0.64886, 0.59656, 0.015907, 0.18678, 0.78037],
            [-0.16572, -0.45536, 0.64454, 0.6423, -0.077473, 0.13652, 0.7502],
            [0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883]
        ]

        self.original_waypoint_count = len(self.waypoints)

        # ---------------- STATE ----------------
        self.active_waypoints = []
        self.active_waypoint_count = 0
        self.current_wp_idx = 0

        self.pose_received = False
        self.current_pose = None

        self.attached = False
        self.detached = True
        self.service_in_progress = False

        self.manipulation_enabled = False
        self.fruit_waypoints_fetched = False
        self.dock_count = 0

        self.waiting = False
        self.wait_start_time = None

        self.dropoff_pose = [-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008]

        # ---------------- CONTROL ----------------
        self.timer = self.create_timer(0.05, self.control_loop)

        self.position_tolerance = 0.02
        self.rotation_tolerance = 0.05

        self.get_logger().info("UR5 Manipulation Node Initialized")

    # =====================================================
    def wait_for_services(self):
        self.attach_client.wait_for_service()
        self.detach_client.wait_for_service()

    def pose_callback(self, msg):
        self.current_pose = msg
        self.pose_received = True

    # =====================================================
    # DOCK LOGIC
    # =====================================================
    def dock_reached_callback(self, msg):
        if not msg.data:
            return

        self.dock_count += 1
        self.get_logger().info(f"Dock reached → count {self.dock_count}")

        # -------- FIRST DOCK --------
        if self.dock_count == 1:
            self.active_waypoints = self.waypoints
            self.active_waypoint_count = len(self.waypoints)
            self.current_wp_idx = 0
            self.manipulation_enabled = True

        # -------- SECOND DOCK --------
        elif self.dock_count == 2:
            self.active_waypoints = self.waypoints2
            self.active_waypoint_count = len(self.waypoints2)
            self.current_wp_idx = 0

            self.attached = False
            self.detached = True
            self.fruit_waypoints_fetched = False

            self.manipulation_enabled = True

    # =====================================================
    # FRUIT FETCH (MISSION 1 ONLY)
    # =====================================================
    def fetch_fruit_waypoints(self):
        source_frame = 'base_link'
        new_wps = []

        for fruit_id in range(1, MAX_FRUIT_ID + 1):
            try:
                tf = self.tf_buffer.lookup_transform(
                    source_frame, f'{TEAM_ID}_bad_fruit_{fruit_id}', rclpy.time.Time())
                t = tf.transform.translation
                r = tf.transform.rotation
                pose = [t.x, t.y, t.z + 0.01, r.x, r.y, r.z, r.w]

                new_wps += [
                    pose,
                    [-0.16, 0.51, 0.63, r.x, r.y, r.z, r.w],
                    self.dropoff_pose,
                    [-0.16, 0.51, 0.63, r.x, r.y, r.z, r.w]
                ]
            except TransformException:
                continue

        self.waypoints.extend(new_wps)
        self.active_waypoints = self.waypoints
        self.active_waypoint_count = len(self.waypoints)
        self.fruit_waypoints_fetched = True

    # =====================================================
    # CONTROL LOOP
    # =====================================================
    def control_loop(self):
        if not self.pose_received or not self.manipulation_enabled:
            return

        if self.current_wp_idx >= self.active_waypoint_count:
            self.twist_pub.publish(Twist())
            self.manipulation_enabled = False
            return

        if self.service_in_progress:
            self.twist_pub.publish(Twist())
            return

        msg = self.current_pose
        cur = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        tgt = self.active_waypoints[self.current_wp_idx]
        twist, pe, re = self.compute_twist_error(cur, tgt)

        if pe < self.position_tolerance and re < self.rotation_tolerance:
            self.handle_service_calls()
            return

        self.twist_pub.publish(twist)

    # =====================================================
    # ATTACH / DETACH LOGIC
    # =====================================================
    def handle_service_calls(self):

        # ===== MISSION 1 (ORIGINAL) =====
        if self.dock_count == 1:

            if self.current_wp_idx == 0 and not self.attached:
                self.call_attach_service('fertiliser_can')
                return

            if self.current_wp_idx == 3 and self.attached:
                self.call_detach_service('fertiliser_can')
                delivered_msg = Bool()
                delivered_msg.data = True
                self.delivery_pub.publish(delivered_msg)
                return

            if self.current_wp_idx == self.original_waypoint_count - 1 \
                    and not self.fruit_waypoints_fetched:
                self.fetch_fruit_waypoints()
                return

            if self.fruit_waypoints_fetched:
                rel = self.current_wp_idx - self.original_waypoint_count
                if rel >= 0:
                    if rel % 4 == 0 and not self.attached:
                        self.call_attach_service('bad_fruit')
                        return
                    if rel % 4 == 2 and self.attached:
                        self.call_detach_service('bad_fruit')
                        return

        # ===== MISSION 2 =====
        if self.dock_count == 2:
            if self.current_wp_idx == 1 and not self.attached:
                self.call_attach_service('fertiliser_can')
                return
            
            if self.current_wp_idx == 2 and self.attached:
                unloaded_msg = Bool()
                unloaded_msg.data = True
                self.unloaded_pub.publish(unloaded_msg)

            if self.current_wp_idx == 5 and self.attached:
                self.call_detach_service('fertiliser_can')
                return

        self.current_wp_idx += 1

    # =====================================================
    def call_attach_service(self, model):
        self.service_in_progress = True
        req = AttachLink.Request()
        req.model1_name = model
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        future = self.attach_client.call_async(req)
        future.add_done_callback(self.attach_done)

    def attach_done(self, future):
        self.service_in_progress = False
        self.attached = True
        self.detached = False
        self.current_wp_idx += 1

    def call_detach_service(self, model):
        self.service_in_progress = True
        req = DetachLink.Request()
        req.model1_name = model
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        future = self.detach_client.call_async(req)
        future.add_done_callback(self.detach_done)

    def detach_done(self, future):
        self.service_in_progress = False
        self.attached = False
        self.detached = True
        self.current_wp_idx += 1

    # =====================================================
    def compute_twist_error(self, cur, tgt):
        pos_err = tgt[:3] - cur[:3]
        r_err = (Rotation.from_quat(tgt[3:]) *
                 Rotation.from_quat(cur[3:]).inv()).as_rotvec()

        twist = Twist()
        twist.linear.x = 6.0 * pos_err[0]
        twist.linear.y = 6.0 * pos_err[1]
        twist.linear.z = 6.0 * pos_err[2]
        twist.angular.x = 5.0 * r_err[0]
        twist.angular.y = 5.0 * r_err[1]
        twist.angular.z = 5.0 * r_err[2]

        return twist, np.linalg.norm(pos_err), np.linalg.norm(r_err)


def main(args=None):
    rclpy.init(args=args)
    node = UR5_Manipulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
