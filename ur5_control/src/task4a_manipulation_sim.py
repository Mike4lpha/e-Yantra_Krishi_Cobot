#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from scipy.spatial.transform import Rotation
import tf2_ros
from tf2_ros import TransformException
import cv2
import signal
from sensor_msgs.msg import JointState
from control_msgs.msg import JointJog
from std_msgs.msg import Float64MultiArray

from linkattacher_msgs.srv import AttachLink, DetachLink

TEAM_ID = "2345" 
MAX_FRUIT_ID = 3

class UR5_Manipulation(Node):
    def __init__(self):
        super().__init__('ur5_manipulation_node')

        self.pose_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        # self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)

        # Service clients
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        # Wait for services to be available
        self.wait_for_services()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.waypoints = [
            # [0.16055, -0.16567, 0.57473,0.707,0.028,0.034,0.707],
            # [0.17339, -0.29082, 0.62107, 0.69210, 0.00733, 0.00386, 0.72175],
            # [0.69968,-0.0052498,0.43932,-0.70134,0.71253,0.012449,-0.016744]
            # [-0.21661,-0.53397,0.62345,0.69210, 0.00733, 0.00386, 0.72175], ## fertilizer
            # # [-0.14, -0.487, 0.668, 0.69210, 0.00733, 0.00386, 0.72175],
            # # [-0.0056316, -0.29841, 0.61116, 0.57286, 0.31563, 0.28677, 0.69998],
            # # -0.079396; -0.32296; 0.57759, 0.62796; -0.031387; -0.027777; 0.77712
            # # -0.0071762; -0.26306; 0.55535, 0.62647; 0.1567; 0.15279; 0.74809
            # [0.032102, -0.23049, 0.54328, 0.59939, 0.28113, 0.27379, 0.69766],
            # # [0.059494, -0.20775, 0.53485, 0.54984, 0.40088, 0.39021, 0.62025],
            # [0.087249, -0.12019, 0.54119, 0.55035, 0.5781, 0.36223, 0.48136],
            # [0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883],
            # #  0.12203; -0.12541; 0.53115, 0.6435; 0.67678; 0.2014; 0.29549
            # [0.31798, -0.1212, 0.49393, 0.61241, 0.79022, -0.022255, -0.0023674],
            # # [0.15039, -0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453]
            # [0.69968,-0.0052498,0.43932,-0.70134,0.71253,0.012449,-0.016744], ## fertilizer drop off
            # [-0.16, 0.51, 0.63, 0.70141, 0.71244, 0.01656, -0.01309]

            # [0.69968,-0.0052498,0.33932,-0.70134,0.71253,0.012449,-0.016744],
            # [0.69968,-0.0052498,0.43932,-0.70134,0.71253,0.012449,-0.016744],
            # [-0.165, 0.532, -0.012+0.4, 0.70141, 0.71244, 0.01656, -0.01309], # bad_fruit_1
            # [-0.16, 0.51, 0.63, 0.70141, 0.71244, 0.01656, -0.01309],
            # [-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008], # drop-off
            # [-0.16, 0.51, 0.63, 0.70141, 0.71244, 0.01656, -0.01309],
            # [-0.049, 0.670, -0.049+0.4, 0.70141, 0.71244, 0.01656, -0.01309], # bad_fruit_2
            # [-0.16, 0.51, 0.63, 0.70141, 0.71244, 0.01656, -0.01309],
            # [-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008], # drop-off
            # [-0.16, 0.51, 0.63, 0.70141, 0.71244, 0.01656, -0.01309],
        ]

        self.original_waypoint_count = len(self.waypoints)
        
        self.current_wp_idx = 0
        self.pose_received = False
        self.current_pose = None
        self.waiting = False
        self.wait_start_time = None

        self.wrist_done = False
        self.wrist_target = -1.67   # desired wrist position
        self.wrist_tol = 0.02


        self.fruit_waypoints_fetched = False
        self.obj3_waypoint_added = False
        self.dropoff_pose = [-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008]

        self.attached = False
        self.detached = True 
        self.service_in_progress = False 

        # Timer
        self.timer = self.create_timer(0.02, self.control_loop)

        # Tolerances
        self.position_tolerance = 0.02  
        self.rotation_tolerance = 0.05 

        self.get_logger().info("Manipulation node initialized.")
        self.get_logger().info(f"Loaded {self.original_waypoint_count} pre-defined waypoints.")
        self.get_logger().info("Waiting for first pose message...")


    def wait_for_services(self):
        self.get_logger().info('Waiting for attach/detach services...')
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /attach_link...')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /detach_link...')
        self.get_logger().info('Attach/Detach services are available.')

    def pose_callback(self, msg):
        self.current_pose = msg
        self.pose_received = True

    def joint_state_callback(self, msg):
        self.positions = msg.position
        self.velocities = msg.velocity
        self.wrist_pos = self.positions[3]

    def wrist_control(self):
        Kp = 20.0
        speed = Kp * (self.wrist_target - self.wrist_pos)
        speed = np.clip(speed, -3.0, 3.0)

        cmd = Float64MultiArray()
        cmd.data = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
        self.joint_pub.publish(cmd)

    def fetch_obj3_waypoint(self):
        self.get_logger().info("--- Checking for fertilizer... ---")
        source_frame = 'base_link'
        try:
            tf = self.tf_buffer.lookup_transform(
                source_frame, '2345_fertilizer_1', rclpy.time.Time()
            )
            t = tf.transform.translation
            r = tf.transform.rotation

            pose = [t.x, t.y + 0.02, t.z, r.x, r.y, r.z, r.w]
            self.waypoints.append(pose)
            self.waypoints.append([0.032102, -0.23049, 0.54328, 0.59939, 0.28113, 0.27379, 0.69766])
            self.waypoints.append([0.087249, -0.12019, 0.54119, 0.55035, 0.5781, 0.36223, 0.48136])
            self.waypoints.append([0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883])
            self.waypoints.append([0.31798, -0.1212, 0.49393, 0.61241, 0.79022, -0.022255, -0.0023674])
            self.waypoints.append([0.69968,-0.0052498,0.43932,-0.70134,0.71253,0.012449,-0.016744])
            self.waypoints.append([-0.16, 0.51, 0.63, 0.70141, 0.71244, 0.01656, -0.01309])

            self.obj3_waypoint_added = True

            self.get_logger().info("Added waypoints")

        except TransformException:
            self.get_logger().warn("fertilizer TF not available")
    
    def fetch_fruit_waypoints(self):
        """
        Queries TF to find all bad fruit and APPENDS a pick-and-place plan
        to the existing waypoint list.
        """
        self.get_logger().info("--- Checking for all bad fruits... ---")
        source_frame = 'base_link'
        
        found_fruit_poses = []
        for fruit_id in range(1, MAX_FRUIT_ID + 1):
            target_frame = f'{TEAM_ID}_bad_fruit_{fruit_id}'
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    source_frame, target_frame, rclpy.time.Time()
                )
                t = transform_stamped.transform.translation
                r = transform_stamped.transform.rotation
                fruit_pose = [t.x, t.y, t.z+0.005, r.x, r.y, r.z, r.w]
                found_fruit_poses.append(fruit_pose)
                self.get_logger().info(
                    f"Found Fruit {fruit_id}: Pos({t.x:.2f}, {t.y:.2f}, {t.z:.2f})"
                )
            except TransformException as ex:
                continue

        if not found_fruit_poses:
            self.get_logger().warn("No bad fruits currently detected in TF.")
        else:
            # --- Build the pick-and-place plan ---
            new_pick_place_waypoints = []
            for pose in found_fruit_poses:
                new_pick_place_waypoints.append(pose)                 # 1. Add waypoint TO the fruit
                new_pick_place_waypoints.append([-0.16, 0.51, 0.63, pose[3], pose[4], pose[5], pose[6]]) # Above fruit
                new_pick_place_waypoints.append(self.dropoff_pose)    # 2. Add waypoint TO the drop-off --> This is detach point
                new_pick_place_waypoints.append([-0.16, 0.51, 0.63, pose[3], pose[4], pose[5], pose[6]]) # Above drop-off

            self.waypoints.extend(new_pick_place_waypoints)
            
            self.get_logger().info(f"Added {len(new_pick_place_waypoints)} new waypoints for {len(found_fruit_poses)} fruits.")
            self.get_logger().info(f"Total waypoints in plan: {len(self.waypoints)}")

        self.fruit_waypoints_fetched = True

    # def show_velocity_monitor(self, lin, ang):
    #     """
    #     lin, ang: np.array([vx, vy, vz]), np.array([wx, wy, wz])
    #     """
    #     h, w = 300, 600
    #     img = np.zeros((h, w, 3), dtype=np.uint8)

    #     def draw_bar(x, y, value, label, scale=5.0, color=(0, 255, 0)):
    #         max_len = 200
    #         length = int(np.clip(value * scale, -1.0, 1.0) * max_len)

    #         # center line
    #         cv2.line(img, (x, y), (x + max_len, y), (100, 100, 100), 1)
    #         cv2.line(img, (x, y), (x - max_len, y), (100, 100, 100), 1)

    #         # value bar
    #         cv2.line(img, (x, y), (x + length, y), color, 6)

    #         cv2.putText(
    #             img,
    #             f"{label}: {value:.3f}",
    #             (x + 220, y + 5),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (255, 255, 255),
    #             1,
    #         )

    #     # Linear velocities (green)
    #     draw_bar(300, 50,  lin[0], "Vx")
    #     draw_bar(300, 90,  lin[1], "Vy")
    #     draw_bar(300, 130, lin[2], "Vz")

    #     # Angular velocities (red)
    #     draw_bar(300, 190, ang[0], "Wx", color=(0, 0, 255))
    #     draw_bar(300, 230, ang[1], "Wy", color=(0, 0, 255))
    #     draw_bar(300, 270, ang[2], "Wz", color=(0, 0, 255))

    #     cv2.imshow("Twist Velocity Monitor (Before Clipping)", img)
    #     cv2.waitKey(1)

    def compute_twist_error(self, current_pose, target_pose, waypoint_index):
        pos_err = np.array(target_pose[:3]) - np.array(current_pose[:3])
        r_current = Rotation.from_quat(current_pose[3:])
        r_target = Rotation.from_quat(target_pose[3:])
        rot_error = (r_target * r_current.inv()).as_rotvec()

        pos_err_norm = np.linalg.norm(pos_err)

        Kp_pos = 0.4
        Kp_rot = 0.3

        twist = Twist()
        twist.linear.x = Kp_pos * pos_err[0]
        twist.linear.y = Kp_pos * pos_err[1]
        twist.linear.z = Kp_pos * pos_err[2]
        twist.angular.x = Kp_rot * rot_error[0]
        twist.angular.y = Kp_rot * rot_error[1]
        twist.angular.z = Kp_rot * rot_error[2]

        twist.linear.x = np.clip(twist.linear.x, -0.1, 0.1)
        twist.linear.y = np.clip(twist.linear.y, -0.1, 0.1)
        twist.linear.z = np.clip(twist.linear.z, -0.1, 0.1)

        twist.angular.x = np.clip(twist.angular.x, -0.3, 0.3)
        twist.angular.y = np.clip(twist.angular.y, -0.3, 0.3)
        twist.angular.z = np.clip(twist.angular.z, -0.3, 0.3)

        # if pos_err_norm > 0.03:
        #     twist.angular.x = 0.0
        #     twist.angular.y = 0.0
        #     twist.angular.z = 0.0

        # lin_vel = np.array([
        #     twist.linear.x,
        #     twist.linear.y,
        #     twist.linear.z
        # ])

        # ang_vel = np.array([
        #     twist.angular.x,
        #     twist.angular.y,
        #     twist.angular.z
        # ])

        # self.show_velocity_monitor(lin_vel, ang_vel)

        return twist, np.linalg.norm(pos_err), np.linalg.norm(rot_error)

    def control_loop(self):
        if not self.pose_received:
            return
        
        if not self.wrist_done:
            if not hasattr(self, 'wrist_pos'):
                return

            self.wrist_control()

            if abs(self.wrist_pos - self.wrist_target) < self.wrist_tol:
                self.get_logger().info("✅ Wrist aligned. Proceeding to fruit task.")
                self.wrist_done = True
                self.twist_pub.publish(Twist())

                if not self.fruit_waypoints_fetched:
                    self.fetch_fruit_waypoints()

            return

        # --- Check if all waypoints done ---
        if self.current_wp_idx >= len(self.waypoints):
            if self.current_wp_idx == len(self.waypoints) and self.fruit_waypoints_fetched:
                self.get_logger().info("All waypoints completed. Shutting down.")
                self.twist_pub.publish(Twist())
                self.current_wp_idx += 1
            return

        # --- Pause control while a service is active ---
        if self.service_in_progress:
            self.twist_pub.publish(Twist())
            return

        # --- Get current pose and orientation ---
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
            if (self.get_clock().now().nanoseconds - self.wait_start_time) / 1e9 >= 0.3:
                self.waiting = False
                self.current_wp_idx += 1
                if self.current_wp_idx < len(self.waypoints):
                    self.get_logger().info(
                        f"Wait complete. Moving to waypoint {self.current_wp_idx + 1}..."
                    )
            else:
                self.twist_pub.publish(Twist())
                return

        twist, pos_err_norm, rot_err_norm = self.compute_twist_error(
            current_pose, target_pose, self.current_wp_idx
        )

        if pos_err_norm < self.position_tolerance and rot_err_norm < self.rotation_tolerance:
            self.get_logger().info(
                f"Reached waypoint {self.current_wp_idx + 1}/{len(self.waypoints)}."
            )

            is_last_original_waypoint = (
                self.current_wp_idx == self.original_waypoint_count - 1
            )

            # ✅ FIRST: fetch obj_3 (replaces old hardcoded waypoints)
            # if is_last_original_waypoint and not self.obj3_waypoint_added:
            #     self.get_logger().info(
            #         "Reached last pre-defined waypoint. Fetching obj_3 waypoints..."
            #     )
            #     self.fetch_obj3_waypoint()
            #     return

            # ✅ THEN: fetch fruits only AFTER obj_3 is reached
            # if (
            #     self.obj3_waypoint_added
            #     and self.current_wp_idx == self.original_waypoint_count and not self.fruit_waypoints_fetched
            # ):
            #     self.get_logger().info(
            #         "Reached fertilizer waypoint. Fetching bad fruit locations..."
            #     )
            #     self.fetch_fruit_waypoints()
            #     return

            # --- Final waypoint handling ---
            if self.current_wp_idx == len(self.waypoints) - 1:
                self.get_logger().info("Reached FINAL waypoint.")
                service_called = self.handle_service_calls()

                if not service_called:
                    self.get_logger().info("All waypoints completed. Shutting down control.")
                    self.twist_pub.publish(Twist())
                    self.current_wp_idx += 1
                return

            # --- Handle services for intermediate waypoints ---
            service_called = self.handle_service_calls()

            if service_called in ["attach", "detach"]:
                self.waiting = True
                self.wait_start_time = self.get_clock().now().nanoseconds
                self.get_logger().info(f"Waiting after {service_called} service...")
                self.twist_pub.publish(Twist())
                return

            if service_called:
                return

            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                self.get_logger().info(
                    f"Moving to waypoint {self.current_wp_idx + 1}..."
                )
            return

        self.twist_pub.publish(twist)
        print(self.positions)

    
    def handle_service_calls(self):
        """
        Checks the current waypoint index and calls attach/detach if needed.
        Returns 'attach', 'detach', or None depending on the service called.
        """

        obj3_start_idx = self.original_waypoint_count
        obj3_detach_idx = obj3_start_idx + 5   # 👈 adjust if needed

        if self.obj3_waypoint_added and self.current_wp_idx == obj3_start_idx and not self.attached:
            self.get_logger().info("Calling ATTACH for fertilizer")
            self.call_attach_service('fertiliser_can')
            return "attach"

        elif self.obj3_waypoint_added and self.current_wp_idx == obj3_detach_idx and self.attached:
            self.get_logger().info("Calling DETACH for fertilizer")
            self.call_detach_service('fertiliser_can')
            return "detach"

        # --- For fruit waypoints ---
        elif self.fruit_waypoints_fetched and self.current_wp_idx >= self.original_waypoint_count:
            relative_fruit_idx = self.current_wp_idx - self.original_waypoint_count


            is_pick_waypoint = (relative_fruit_idx % 4 == 0)
            is_place_waypoint = (relative_fruit_idx % 4 == 2)

            fruit_num = (relative_fruit_idx // 4) + 1

            if is_pick_waypoint and not self.attached:
                self.get_logger().info(f"Calling ATTACH for fruit pick {fruit_num}...")
                self.call_attach_service('bad_fruit')
                return "attach"

            elif is_place_waypoint and self.attached:
                self.get_logger().info(f"Calling DETACH for fruit place {fruit_num}...")
                self.call_detach_service('bad_fruit')
                return "detach"

        return None

    
    def call_attach_service(self, model_name): 
        self.get_logger().info(f"Calling attach service for model: {model_name}...")
        self.service_in_progress = True
        
        req = AttachLink.Request()
        req.model1_name = model_name 
        req.link1_name = 'body'      
        req.model2_name = 'ur5' 
        req.link2_name = 'wrist_3_link'

        future = self.attach_client.call_async(req)
        future.add_done_callback(self.attach_response_callback)

    def attach_response_callback(self, future):
        self.service_in_progress = False
        try:
            result = future.result()
            if result is not None and result.success:
                self.get_logger().info("Attach succeeded.")
                self.attached = True
                self.detached = False
            else:
                self.get_logger().error(f"Attach failed! Result: {result}")
        except Exception as e:
            self.get_logger().error(f"Attach service call exception: {e}")
        
        self.waiting = True
        self.wait_start_time = self.get_clock().now().nanoseconds
            
    def call_detach_service(self, model_name): 
        self.get_logger().info(f"Calling detach service for model: {model_name}...")
        self.service_in_progress = True

        req = DetachLink.Request()
        req.model1_name = model_name 
        req.link1_name = 'body'      
        req.model2_name = 'ur5' 
        req.link2_name = 'wrist_3_link'

        future = self.detach_client.call_async(req)
        future.add_done_callback(self.detach_response_callback)

    def detach_response_callback(self, future):
        self.service_in_progress = False
        try:
            result = future.result()
            if result is not None and result.success:
                self.get_logger().info("Detach succeeded.")
                self.detached = True
                self.attached = False
            else:
                self.get_logger().error(f"Detach failed! Result: {result}")
        except Exception as e:
             self.get_logger().error(f"Detach service call exception: {e}")

        self.waiting = True
        self.wait_start_time = self.get_clock().now().nanoseconds

    def stop_robot(self):
        self.twist_pub.publish(Twist())
        

def main(args=None):
    rclpy.init(args=args)
    node = UR5_Manipulation()

    shutdown_called = False

    def shutdown_handler(sig, frame):
        nonlocal shutdown_called
        if shutdown_called:
            return

        shutdown_called = True
        try:
            node.get_logger().warn(
                f"Shutdown signal {sig} received. Stopping UR5 safely."
            )
            node.stop_robot()
        except Exception:
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()

    # Handle all common termination cases
    signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, shutdown_handler)  # VS Code Stop
    signal.signal(signal.SIGHUP, shutdown_handler)   # Terminal closed

    rclpy.spin(node)


if __name__ == '__main__':
    main()