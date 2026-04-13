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

        self.pose_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # Service clients
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        # Wait for services to be available
        self.wait_for_services()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.waypoints = [
            [-0.21661,-0.53397,0.62345,0.707,0.028,0.034,0.707],
            [0.12039, -0.10902, 0.44477, 0.50075, 0.49696, 0.5034, 0.49883],
            [0.15039, -0.12902, 0.49477, 0.684, 0.726, -0.0271, 0.01453],
            [0.69968,-0.0052498,0.33932,-0.70134,0.71253,0.012449,-0.016744],
            [0.69968,-0.0052498,0.43932,-0.70134,0.71253,0.012449,-0.016744]
        ]
        self.original_waypoint_count = len(self.waypoints)
        
        self.current_wp_idx = 0
        self.pose_received = False
        self.current_pose = None
        self.waiting = False
        self.wait_start_time = None

        self.fruit_waypoints_fetched = False

        self.dropoff_pose = [-0.806, 0.010, 0.182, -0.684, 0.726, 0.05, 0.008]

        self.attached = False
        self.detached = True 
        self.service_in_progress = False 

        # Timer
        self.timer = self.create_timer(0.05, self.control_loop)

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
                fruit_pose = [t.x, t.y, t.z+0.01, r.x, r.y, r.z, r.w]
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

    def compute_twist_error(self, current_pose, target_pose, waypoint_index):
        pos_err = np.array(target_pose[:3]) - np.array(current_pose[:3])
        r_current = Rotation.from_quat(current_pose[3:])
        r_target = Rotation.from_quat(target_pose[3:])
        rot_error = (r_target * r_current.inv()).as_rotvec()

        Kp_pos = 7.0
        Kp_rot = 6.0

        twist = Twist()
        twist.linear.x = Kp_pos * pos_err[0]
        twist.linear.y = Kp_pos * pos_err[1]
        twist.linear.z = Kp_pos * pos_err[2]
        twist.angular.x = Kp_rot * rot_error[0]
        twist.angular.y = Kp_rot * rot_error[1]
        twist.angular.z = Kp_rot * rot_error[2]

        return twist, np.linalg.norm(pos_err), np.linalg.norm(rot_error)

    def control_loop(self):
        if not self.pose_received:
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
                    self.get_logger().info(f"Wait complete. Moving to waypoint {self.current_wp_idx + 1}...")
            else:
                self.twist_pub.publish(Twist())  
                return
            
        twist, pos_err_norm, rot_err_norm = self.compute_twist_error(current_pose, target_pose, self.current_wp_idx)

        if pos_err_norm < self.position_tolerance and rot_err_norm < self.rotation_tolerance:
            self.get_logger().info(f"Reached waypoint {self.current_wp_idx + 1}/{len(self.waypoints)}.")

            is_last_original_waypoint = (self.current_wp_idx == self.original_waypoint_count - 1)

            # --- Fetch fruit waypoints after last original one ---
            if is_last_original_waypoint and not self.fruit_waypoints_fetched:
                self.get_logger().info("Reached last pre-defined waypoint. Fetching bad fruit locations...")
                self.fetch_fruit_waypoints()
                return

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

            # Wait only for attach/detach services
            if service_called in ["attach", "detach"]:
                self.waiting = True
                self.wait_start_time = self.get_clock().now().nanoseconds
                self.get_logger().info(f"Waiting after {service_called} service...")
                self.twist_pub.publish(Twist())  # stop motion while waiting
                return

            # Other services (like detect) just proceed immediately
            if service_called:
                return

            # No service -> move to next waypoint directly
            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                self.get_logger().info(f"Moving to waypoint {self.current_wp_idx + 1}...")
            return

        self.twist_pub.publish(twist)
    
    def handle_service_calls(self):
        """
        Checks the current waypoint index and calls attach/detach if needed.
        Returns 'attach', 'detach', or None depending on the service called.
        """
        # --- For original waypoints ---
        if self.current_wp_idx < self.original_waypoint_count:
            if self.current_wp_idx == 0 and not self.attached:
                self.get_logger().info("Calling ATTACH for original waypoint 1...")
                self.call_attach_service('fertiliser_can')
                return "attach"
            
            elif self.current_wp_idx == 3 and not self.detached:
                self.get_logger().info("Calling DETACH for original waypoint 4...")
                self.call_detach_service('fertiliser_can')
                return "detach"

        # --- For fruit waypoints ---
        elif self.fruit_waypoints_fetched:
            relative_fruit_idx = self.current_wp_idx - self.original_waypoint_count
            
            # Waypoint index pattern:
            # 0: Pick Fruit 1
            # 1: Above Fruit 1
            # 2: Place Fruit 1
            # 3: Above Place 1
            # 4: Pick Fruit 2
            # ...
            is_pick_waypoint = (relative_fruit_idx % 4 == 0)
            place_indices = [2, 6, 10]  # or computed dynamically if needed
            is_place_waypoint = (relative_fruit_idx in place_indices)
            
            fruit_num = (relative_fruit_idx // 4) + 1

            if is_pick_waypoint and not self.attached:
                self.get_logger().info(f"Calling ATTACH for bad fruit {fruit_num} pick...")
                self.call_attach_service('bad_fruit')
                return "attach"
            
            elif is_place_waypoint and self.attached:
                self.get_logger().info(f"Calling DETACH for bad fruit {fruit_num} place...")
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


def main(args=None):
    rclpy.init(args=args)
    node = UR5_Manipulation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node interrupted, shutting down cleanly.")
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()