#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
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

        self.shape_detected_left = False
        self.shape_detected_right = False
        self.detected_shape = None
        self.shape_detected_time = None

        self.pose = None
        self.y_ok = False 
        self.ignore_zone = False
        self.status = "Unknown"
        self.flag = False
        self.plant_no = 1
        self.stop_detection = False
        self.max_shapes = 5

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

        self.y_ok = -5.0 <= y <= 0.81
        self.ignore_zone1 = -5.44 < y < 1.35 and 0.30 < x < 0.45
        self.ignore_zone3 = -5.63 < y < 1.31 and -3.60 < x < -3.30

        
    def stop_detection_callback(self, msg):
        self.stop_detection = msg.data
        if self.stop_detection:
            self.get_logger().info("Shape detection DISABLED.")
        
    def scan_callback(self, msg):

        if self.stop_detection:
            return 

        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]

        left_mask = (angles > np.deg2rad(60)) & (angles < np.deg2rad(120))
        right_mask = (angles < np.deg2rad(-60)) & (angles > np.deg2rad(-120))

        left_points = self.ranges_to_points(ranges[left_mask], angles[left_mask])
        right_points = self.ranges_to_points(ranges[right_mask], angles[right_mask])

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

        pts_img = np.round(points * scale + img_size // 2).astype(int)
        pts_img = pts_img[(pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_size)
                          & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_size)]

        for (px, py) in pts_img:
            img[py, px] = (255, 255, 255)

        num_lines = 0
        shape = "Unknown"
        angle = "N/A"
        
        if self.y_ok and not self.ignore_zone3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=2, maxLineGap=100)
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

            self.get_logger().info("Shape Detected...Robot Stopped.")

            # x = self.pose[0] 
            # y = self.pose[1] 

            if shape == "Square":
                self.status = "BAD_HEALTH"
            elif shape == "Triangle":
                self.status = "FERTILIZER_REQUIRED"

            # msg = String()
            # msg.data = f"{status},{x:.2f},{y:.2f}"
            # self.detection_pub.publish(msg)
            # self.get_logger().info(f"Detection published: {msg.data}")

        cv2.putText(img, f"Lines: {num_lines}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if shape in ["Square", "Triangle"]:
            cv2.putText(img, f"Shape: {shape}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Angle: {angle}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # cv2.imshow(window_name, img)
        # cv2.waitKey(1)

    def process_right_side(self, points, window_name):
        
        img_size = 500
        scale = 75
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        pts_img = np.round(points * scale + img_size // 2).astype(int)
        pts_img = pts_img[(pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_size)
                          & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_size)]

        for (px, py) in pts_img:
            img[py, px] = (255, 255, 255)

        num_lines = 0
        shape = "Unknown"
        angle = "N/A"
        
        if self.y_ok and not self.ignore_zone1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=2, maxLineGap=100)
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

            self.get_logger().info("Robot stopping due to shape detection.")

            # x = self.pose[0] 
            # y = self.pose[1] 

            if shape == "Square":
                self.status = "BAD_HEALTH"
            elif shape == "Triangle":
                self.status = "FERTILIZER_REQUIRED"

            # msg = String()
            # msg.data = f"{status},{x:.2f},{y:.2f}"
            # self.detection_pub.publish(msg)
            # self.get_logger().info(f"Detection published: {msg.data}")

        cv2.putText(img, f"Lines: {num_lines}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if shape in ["Square", "Triangle"]:
            cv2.putText(img, f"Shape: {shape}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Angle: {angle}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # cv2.imshow(window_name, img)
        # cv2.waitKey(1)

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
        
        unique_dirs = []
        if len(angles) > 0:
            unique_dirs.append(angles[0])
            for a in angles[1:]:
                diffs = [abs(a - u) for u in unique_dirs]
                min_diff = min(diffs + [np.pi - d for d in diffs]) 
                
                if min_diff > ANGLE_TOLERANCE_RAD:
                    unique_dirs.append(a)

        sides = len(unique_dirs)
        
        if sides == 1:
            return "Line", "N/A"
        
        elif sides == 2:
            angle_diff = abs(unique_dirs[0] - unique_dirs[1])
            angle_between = min(angle_diff, np.pi - angle_diff)
            angle_deg = np.rad2deg(angle_between)
            angle_str = f"{angle_deg:.1f} deg"
            
            if angle_deg < ANGLE_TOLERANCE_DEG:
                return "Line", angle_str
            
            elif abs(angle_deg - 90) < ANGLE_TOLERANCE_DEG:
                return "Square", angle_str
            
            elif abs(angle_deg - 40) < ANGLE_TOLERANCE_DEG:
                return "Triangle", angle_str
            
            else:
                return "Corner", angle_str
        
        elif sides > 2: 
            return "Unknown", f"{sides} sides"
        
        else:
            return "Unknown", "N/A"

    def merge_lines(self, lines, angle_thresh_rad=np.deg2rad(10), rho_thresh_px=30):
        if lines is None:
            return None

        line_params = [] 
        for line_seg in lines:
            x1, y1, x2, y2 = line_seg[0]
            angle = math.atan2(y2 - y1, x2 - x1)
            rho = x1 * math.cos(angle) + y1 * math.sin(angle)
            if angle < 0:
                angle += np.pi
                if rho != 0:
                    rho = -rho
            line_params.append({'rho': rho, 'angle': angle, 'points': [(x1,y1), (x2,y2)]})

        groups = []
        for line in line_params:
            found_group = False
            for group in groups:
                avg_rho = group['avg_rho']
                avg_angle = group['avg_angle']
                angle_diff = abs(line['angle'] - avg_angle)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                if angle_diff < angle_thresh_rad and abs(line['rho'] - avg_rho) < rho_thresh_px:
                    group['lines'].append(line)
                    group['avg_rho'] = (group['avg_rho'] * (len(group['lines']) - 1) + line['rho']) / len(group['lines'])
                    group['avg_angle'] = (group['avg_angle'] * (len(group['lines']) - 1) + line['angle']) / len(group['lines'])
                    found_group = True
                    break
            if not found_group:
                groups.append({
                    'lines': [line],
                    'avg_rho': line['rho'],
                    'avg_angle': line['angle']
                })

        merged_lines_out = []
        for group in groups:
            all_points = []
            for line in group['lines']:
                all_points.extend(line['points'])
            if not all_points:
                continue
            all_points_np = np.array(all_points)
            line_fit = cv2.fitLine(all_points_np, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line_fit.flatten()
            t = (all_points_np[:, 0] - x0) * vx + (all_points_np[:, 1] - y0) * vy
            t_min, t_max = np.min(t), np.max(t)
            x1 = int(round(x0 + t_min * vx))
            y1 = int(round(y0 + t_min * vy))
            x2 = int(round(x0 + t_max * vx))
            y2 = int(round(y0 + t_max * vy))
            merged_lines_out.append([[x1, y1, x2, y2]]) 

        return np.array(merged_lines_out) if merged_lines_out else None

    def check_reset_condition(self):
        if self.detected_shape and self.shape_detected_time is not None:
            elapsed_time = (self.get_clock().now() - self.shape_detected_time).nanoseconds / 1e9
            if elapsed_time > 5.5:
                self.get_logger().info(f"Resetting shape detection after {elapsed_time:.1f} seconds.")
                self.reset_shape_detection()

    def check_publish_odom(self):
        if self.detected_shape and self.shape_detected_time is not None:
            elapsed_time = (self.get_clock().now() - self.shape_detected_time).nanoseconds / 1e9
            if elapsed_time > 2.2 and not self.flag:
                self.get_logger().info("Publishing odom")
                self.publish_odom()
                
    def plant_id(self, y):
        if self.shape_detected_left:
            if -4.819289211553887 <= y <= -3.6229623154486656:
                self.plant_ID = 1

            if -3.6229623154486656 <= y <= -2.2313851184473132:
                self.plant_ID = 2

            if -2.2313851184473132 <= y <= -0.8116727505546906:
                self.plant_ID = 3

            if -0.8116727505546906 <= y <= 0.6157087372467281:
                self.plant_ID = 4
        if self.shape_detected_right:
            if -4.819289211553887 <= y <= -3.6229623154486656:
                self.plant_ID = 5

            if -3.6229623154486656 <= y <= -2.2313851184473132:
                self.plant_ID = 6

            if -2.2313851184473132 <= y <= -0.8116727505546906:
                self.plant_ID = 7

            if -0.8116727505546906 <= y <= 0.6157087372467281:
                self.plant_ID = 8
    def publish_odom(self):
        x = self.pose[0]
        y = self.pose[1]

        self.plant_id(y)

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
