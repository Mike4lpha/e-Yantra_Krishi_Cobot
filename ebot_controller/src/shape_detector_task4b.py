#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion
import numpy as np
import cv2
import math
import time

class ShapeDetection(Node):
    def __init__(self):
        super().__init__('shape_detection_hough_node')

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.create_subscription(Bool, '/stop_detection', self.stop_detection_callback, 10)
        self.bridge = CvBridge()
        self.image_pub_left = self.create_publisher(Image, '/shape_detection/left_image', 10)
        self.image_pub_right = self.create_publisher(Image, '/shape_detection/right_image', 10)


        self.shape_detected_left = False
        self.shape_detected_right = False
        self.detected_shape = None
        self.shape_detected_time = None

        self.pose = None
        self.x_ok = False
        # self.x_ok = True
        self.ignore_zone = False
        # self.ignore_zone1 = False
        # self.ignore_zone3 = False
        self.status = "Unknown"
        self.flag = False
        self.plant_no = 1
        self.stop_detection = False
        self.max_shapes = 3

        self.timer = self.create_timer(0.5, self.check_reset_condition)
        self.timer = self.create_timer(0.5, self.check_publish_odom)

        self.get_logger().info("Shape detection node initialized.")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])
        self.pose = [pos.x, pos.y, yaw]

        x = self.pose[0]
        y = self.pose[1]

        self.x_ok = 1.68 <= x <= 3.60
        self.ignore_zone1 = -2.330 < y < -1.144  and 0.60 < x < 4.75 ## 4.90
        self.ignore_zone3 = 1.00 < y < 2.27 and 0.60 < x < 4.75


    def stop_detection_callback(self, msg):
        self.stop_detection = msg.data
        if self.stop_detection:
            self.get_logger().info("Shape detection DISABLED.")

    def remove_outward_notches_radial(self, ranges, angles, window_size=5):
        """
        Removes small outward radial spikes w.r.t odom using median filtering.
        Keeps true object boundaries intact.
        """
        if len(ranges) < window_size:
            return ranges

        filtered = ranges.copy()
        half = window_size // 2

        for i in range(half, len(ranges) - half):
            local = ranges[i - half : i + half + 1]
            median = np.median(local)

            # remove only outward spikes
            if ranges[i] > median * 1.15:
                filtered[i] = median

        return filtered

    def scan_callback(self, msg):

        if self.stop_detection:
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]

        left_mask = (angles > np.deg2rad(45)) & (angles < np.deg2rad(90))
        right_mask = (angles < np.deg2rad(-45)) & (angles > np.deg2rad(-90))

        # left_points = self.ranges_to_points(ranges[left_mask], angles[left_mask])
        # right_points = self.ranges_to_points(ranges[right_mask], angles[right_mask])
        left_ranges = ranges[left_mask]
        left_angles = angles[left_mask]

        right_ranges = ranges[right_mask]
        right_angles = angles[right_mask]

        # REMOVE OUTWARD NOTCHES (KEY FIX)
        left_ranges = self.remove_outward_notches_radial(left_ranges, left_angles)
        right_ranges = self.remove_outward_notches_radial(right_ranges, right_angles)

        left_points = self.ranges_to_points(left_ranges, left_angles)
        right_points = self.ranges_to_points(right_ranges, right_angles)

        if self.plant_no > self.max_shapes:
            return

        if not self.shape_detected_left:
            self.process_left_side(left_points, "Left Side")

        if not self.shape_detected_right:
            self.process_right_side(right_points, "Right Side")

    def ranges_to_points(self, ranges, angles):
        if len(ranges) == 0:
            return np.empty((0, 2))
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.stack((x, y), axis=1)

    def process_left_side(self, points, window_name):
        img_size = 500
        scale = 75
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # >>> ADDED (ONLY CHANGE)
        # points = self.remove_inward_notches(points)

        pts_img = np.round(points * scale + img_size // 2).astype(int)
        pts_img = pts_img[
            (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_size) &
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_size)
        ]

        for (px, py) in pts_img:
            img[py, px] = (255, 255, 255)

        num_lines = 0
        shape = "Unknown"
        angle = "N/A"

        if self.x_ok and not self.ignore_zone3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            line_segments = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=10, minLineLength=2, maxLineGap=100
            )
            merged_lines = self.merge_lines(line_segments)

            if merged_lines is not None:
                num_lines = len(merged_lines)
                for line in merged_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                shape, angle = self.classify_shape(merged_lines)

        if shape in ["Square", "Triangle"] and not self.shape_detected_left:
            self.shape_detected_left = True
            self.detected_shape = shape
            self.shape_detected_time = self.get_clock().now()

            if shape == "Square":
                self.status = "BAD_HEALTH"
            elif shape == "Triangle":
                self.status = "FERTILIZER_REQUIRED"

        cv2.putText(img, f"Lines: {num_lines}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if shape in ["Square", "Triangle"]:
            cv2.putText(img, f"Shape: {shape}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Angle: {angle}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow(window_name, img)
        cv2.waitKey(1)

        self.image_pub_left.publish(
            self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        )


    def process_right_side(self, points, window_name):
        img_size = 500
        scale = 75
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # >>> ADDED (ONLY CHANGE)
        # points = self.remove_inward_notches(points)

        pts_img = np.round(points * scale + img_size // 2).astype(int)
        pts_img = pts_img[
            (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_size) &
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_size)
        ]

        for (px, py) in pts_img:
            img[py, px] = (255, 255, 255)

        num_lines = 0
        shape = "Unknown"
        angle = "N/A"

        if self.x_ok and not self.ignore_zone1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            line_segments = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=10, minLineLength=2, maxLineGap=100
            )
            merged_lines = self.merge_lines(line_segments)

            if merged_lines is not None:
                num_lines = len(merged_lines)
                for line in merged_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                shape, angle = self.classify_shape(merged_lines)

        if shape in ["Square", "Triangle"] and not self.shape_detected_right:
            self.shape_detected_right = True
            self.detected_shape = shape
            self.shape_detected_time = self.get_clock().now()

            if shape == "Square":
                self.status = "BAD_HEALTH"
            elif shape == "Triangle":
                self.status = "FERTILIZER_REQUIRED"

        cv2.putText(img, f"Lines: {num_lines}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if shape in ["Square", "Triangle"]:
            cv2.putText(img, f"Shape: {shape}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Angle: {angle}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow(window_name, img)
        cv2.waitKey(1)

        self.image_pub_right.publish(
            self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        )


    def classify_shape(self, lines):
        ANGLE_TOLERANCE_DEG = 5.0
        ANGLE_TOLERANCE_RAD = np.deg2rad(ANGLE_TOLERANCE_DEG)

        if lines is None or len(lines) == 0:
            return "Unknown", "N/A"

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1)
            if angle < 0:
                angle += np.pi
            angles.append(angle)

        angles = np.sort(np.array(angles))
        unique_dirs = [angles[0]]

        for a in angles[1:]:
            diffs = [abs(a - u) for u in unique_dirs]
            if min(diffs + [np.pi - d for d in diffs]) > ANGLE_TOLERANCE_RAD:
                unique_dirs.append(a)

        sides = len(unique_dirs)

        if sides == 2:
            diff = abs(unique_dirs[0] - unique_dirs[1])
            diff = min(diff, np.pi - diff)
            deg = np.rad2deg(diff)

            if abs(deg - 90) < ANGLE_TOLERANCE_DEG:
                return "Triangle", f"{deg:.1f} deg"
            if abs(deg - 40) < ANGLE_TOLERANCE_DEG:
                return "Square", f"{deg:.1f} deg"

        return "Unknown", "N/A"

    def merge_lines(self, lines, angle_thresh_rad=np.deg2rad(10), rho_thresh_px=30):
        if lines is None:
            return None

        merged = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if np.hypot(x2 - x1, y2 - y1) > 30:
                merged.append(l)

        return np.array(merged) if merged else None

    def check_reset_condition(self):
        if self.detected_shape and self.shape_detected_time is not None:
            elapsed_time = (self.get_clock().now() - self.shape_detected_time).nanoseconds / 1e9
            if elapsed_time > 5.0:
                self.get_logger().info(f"Resetting shape detection after {elapsed_time:.1f} seconds.")
                self.reset_shape_detection()

    def check_publish_odom(self):
        if self.detected_shape and self.shape_detected_time is not None:
            elapsed_time = (self.get_clock().now() - self.shape_detected_time).nanoseconds / 1e9
            if elapsed_time > 1.0 and not self.flag:
                self.get_logger().info("Publishing odom")
                self.publish_odom()

    def plant_id(self, x):
        if self.shape_detected_left:
            if 1.1814497709274292 <= x <= 1.8315575122833252:
                self.plant_ID = 1

            if 1.8315575122833252 <= x <= 2.5740091800689697:
                self.plant_ID = 2

            if 2.5740091800689697 <= x <= 3.313533306121826:
                self.plant_ID = 3

            if 3.313533306121826 <= x <= 4.04448127746582:
                self.plant_ID = 4

        if self.shape_detected_right:
            if 1.0133718252182007 <= x <= 1.7470312118530273:
                self.plant_ID = 5

            if 1.7470312118530273 <= x <= 2.503039836883545:
                self.plant_ID = 6

            if 2.503039836883545 <= x <= 3.2585418224334717:
                self.plant_ID = 7

            if 3.2585418224334717 <= x <= 3.998225212097168:
                self.plant_ID = 8

    def publish_odom(self):
        x = self.pose[0]
        y = self.pose[1]

        self.plant_id(x)

        msg = String()
        msg.data = f"{self.status},{x:.2f},{y:.2f},{self.plant_ID}"
        self.detection_pub.publish(msg)
        self.get_logger().info(f"Detection published: {msg.data}")
        self.flag = True
        self.plant_no += 1

    def reset_shape_detection(self):
        self.shape_detected_left = False
        self.shape_detected_right = False
        self.detected_shape = None
        self.shape_detected_time = None
        self.flag = False
        self.get_logger().info("Shape detection reset and ready for next shape.")

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
