#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math
import cv2.aruco as aruco
import tf_transformations as tft

# runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False
teamid = "2345"

class FruitsTF(Node):
    
    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.mask = None

        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        # self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        # self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # Publishers
        self.pub_final = self.create_publisher(Image, "/final_image", rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)
        self.pub_crop = self.create_publisher(Image, "/cropped_image", rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT)

        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)


        self.br = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF boilerplate node started.")

    def depthimagecb(self, data):
      
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        cv_image_array = np.array(depth_image, dtype = np.dtype('f8'))
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        self.visualize_depth_image = cv_image_norm
        self.depth_image=depth_image

    def colorimagecb(self, data):
  
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.cv_image=frame


    def bad_fruit_detection(self, rgb_image):

        bad_fruits = []

        y1, y2 = 420, 660
        x1, x2 = 80, 440

        # y1, y2 = 200, 380
        # x1, x2 = 0, 400

        self.cropped_rgb = rgb_image[y1:y2, x1:x2]

        x_offset = x1
        y_offset = y1

        hsv = cv2.cvtColor(self.cropped_rgb, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        lower_pink = np.array([140, 40, 40])
        upper_pink = np.array([179, 255, 255])

        fruit_id = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 300 or area > 20000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cX = x + w // 2 + x_offset
            cY = y + h // 2 + y_offset

            body_y1 = y + int(h // 1.5)
            body_y2 = y + int(1.5 * h)

            body_hsv = hsv[body_y1:body_y2, x:x + w]

            mask_pink_body = cv2.inRange(body_hsv, lower_pink, upper_pink)
            pink_pixels = np.count_nonzero(mask_pink_body)
            total_pixels = mask_pink_body.size
            pink_ratio = pink_pixels / float(total_pixels)
            
            # if pink_ratio > 0.05: 
            #     continue

            distance = float(self.depth_image[cY, cX])

            # angle = np.arctan2((cX - centerCamX), focalX) * (180.0 / np.pi)

            # Store fruit data
            fruit_info = {
                'center': (cX, cY),
                'distance': distance,
                'width': w,
                'id': fruit_id
            }

            bad_fruits.append(fruit_info)
            fruit_id += 1

        return bad_fruits
    
    def aruco_publish_tf(self, marker_id, position):
        # Create a TransformStamped message
        q = TransformStamped()
        q.header.stamp = self.get_clock().now().to_msg()  # Use ROS 2 clock for the timestamp
        q.header.frame_id = "camera_link"
        q.child_frame_id = f"camera_{marker_id}"
        q.transform.translation.x = position[2] 
        q.transform.translation.y = position[0]
        q.transform.translation.z = position[1] 
        q.transform.rotation.x = -2.75
        q.transform.rotation.y = 2.8
        q.transform.rotation.z = -1.0
        q.transform.rotation.w = 1.0

        # Publish the transform
        self.br.sendTransform(q)

        try:
            cam_to_base = self.tf_buffer.lookup_transform('base_link', f'camera_{marker_id}', rclpy.time.Time())
            
            # Publish transform from base_link to bad_fruit
            s = TransformStamped()
            s.header.stamp = self.get_clock().now().to_msg()
            s.header.frame_id = "base_link"
            s.transform.translation.x = cam_to_base.transform.translation.x 
            s.transform.translation.y = cam_to_base.transform.translation.y 
            s.transform.translation.z = cam_to_base.transform.translation.z 
              
            if marker_id == 3:
                s.child_frame_id = f'{teamid}_fertilizer_1'
                q_orig = [
                    cam_to_base.transform.rotation.x,
                    cam_to_base.transform.rotation.y,
                    cam_to_base.transform.rotation.z,
                    cam_to_base.transform.rotation.w,
                ]
                q_z = tft.quaternion_about_axis(math.pi / 2, (0, 0, 1))  
                q_x = tft.quaternion_about_axis(-math.pi / 2, (1, 0, 0))  

                q_extra = tft.quaternion_multiply(q_x, q_z)
                q_final = tft.quaternion_multiply(q_extra, q_orig)
                s.transform.rotation.x, s.transform.rotation.y, s.transform.rotation.z, s.transform.rotation.w = q_final
            
            if marker_id == 6:
                s.child_frame_id = f'obj_{marker_id}'
                q_orig = [
                    cam_to_base.transform.rotation.x,
                    cam_to_base.transform.rotation.y,
                    cam_to_base.transform.rotation.z,
                    cam_to_base.transform.rotation.w,
                ]
                q_extra = tft.quaternion_about_axis(math.pi, (0, 0, 1)) 
                q_final = tft.quaternion_multiply(q_extra, q_orig)
                s.transform.rotation.x, s.transform.rotation.y, s.transform.rotation.z, s.transform.rotation.w = q_final

            
            self.br.sendTransform(s)

            self.get_logger().info(f'Successfully received data!')

        except tf2_ros.TransformException as e:
            self.get_logger().info(f'Could not transform base_link to child_link: {e}')
            return


    def process_image(self):
     
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312
            
        def quaternion_from_euler(ai, aj, ak):
            ai /= 2.0
            aj /= 2.0
            ak /= 2.0
            ci = math.cos(ai)
            si = math.sin(ai)
            cj = math.cos(aj)
            sj = math.sin(aj)
            ck = math.cos(ak)
            sk = math.sin(ak)
            cc = ci*ck
            cs = ci*sk
            sc = si*ck
            ss = si*sk

            q = np.empty((4, ))
            q[0] = cj*sc - sj*cs
            q[1] = cj*ss + sj*cc
            q[2] = cj*cs - sj*sc
            q[3] = cj*cc + sj*ss

            return q
        

        # Detect Aruco Markers and Estimate Pose

        center_aruco_list = []
        distance_from_rgb_list = []
        detected_ids = []
        ids = []

        cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
   
        dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])

        if self.cv_image is None or self.depth_image is None:
            return

        rgb_image = self.cv_image.copy()

        imgGray=cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)

        arucoDict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        arucoParam=aruco.DetectorParameters()

        corners,detected_ids,rejected= aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)

        for i, corner in enumerate(corners):

            aruco.drawDetectedMarkers(rgb_image,corners,detected_ids)

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.01, cam_mat, dist_mat)

            cv2.drawFrameAxes(rgb_image, cam_mat, dist_mat, rvec, tvec, 0.01)
            pts = corner[0]
            center_points = np.mean(pts, axis=0)

            cenX = int(np.round(center_points[0]))
            cenY = int(np.round(center_points[1]))

            cv2.circle(rgb_image, (cenX, cenY), 3, (255,0,255), -1)

            center_aruco_list.append(center_points)
                
            ids.append(detected_ids[i][0])

        for i, marker_id in enumerate(ids):

            cX, cY = center_aruco_list[i]
            cX = int(np.round(cX))
            cY = int(np.round(cY))
           
            height, width = self.depth_image.shape

            if 0 <= cX < width and 0 <= cY < height:
                depth_value_mm = self.depth_image[cY, cX]
                depth_value_m = depth_value_mm / 1000.0 # Convert to meters
            else:
                print(f"Error: cX ({cX}) or cY ({cY}) is out of bounds.")
                continue

            
            x = depth_value_m * (sizeCamX- cX - centerCamX) / focalX
            y = depth_value_m * (sizeCamY- cY - centerCamY) / focalY
            z = depth_value_m 

            position = [float(x),float(y),float(z)]

            self.aruco_publish_tf(marker_id, position)


        # Detect bad fruits
        bad_fruits = self.bad_fruit_detection(rgb_image)

        for fruit in bad_fruits:

            cX = fruit['center'][0]
            cY = fruit['center'][1]
            distance_from_rgb = fruit['distance']
            # angle = fruit['angle']
            fruit_id = fruit['id']
            # print(distance_from_rgb)

            # if depth_value == 0:
            #     continue
            distance_from_rgb = distance_from_rgb / 1000.0
            # print(distance_from_rgb)

            x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
            y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
            z = distance_from_rgb

            # print(x,y,z)

            # # Publish transform from camera_link to bad_fruit
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_link'
            t.child_frame_id = 'cam_' + str(fruit_id)

            t.transform.translation.x = y + 1.65
            t.transform.translation.y = x 
            t.transform.translation.z = z - 1.65

            # q = quaternion_from_euler(0.0 , 0.0, 0.0)
            t.transform.rotation.x = -2.75
            t.transform.rotation.y = 2.8
            t.transform.rotation.z = -1.0
            t.transform.rotation.w = 1.0

            self.br.sendTransform(t)

            # Lookup transform from base_link to camera_link
            try:
                camera_to_base = self.tf_buffer.lookup_transform('base_link', 'cam_' + str(fruit_id), rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f"Cannot get base_link to camera_link transform: {e}")
                continue

            try:
            # Publish transform from base_link to bad_fruit
                obj_transform = TransformStamped()
                obj_transform.header.stamp = self.get_clock().now().to_msg()
                obj_transform.header.frame_id = "base_link"
                obj_transform.child_frame_id = f'{teamid}_bad_fruit_{fruit_id}'
                obj_transform.transform.translation.x = camera_to_base.transform.translation.x 
                obj_transform.transform.translation.y = camera_to_base.transform.translation.y 
                obj_transform.transform.translation.z = camera_to_base.transform.translation.z 
                obj_transform.transform.rotation.x = camera_to_base.transform.rotation.x
                obj_transform.transform.rotation.y = camera_to_base.transform.rotation.y
                obj_transform.transform.rotation.z = camera_to_base.transform.rotation.z
                obj_transform.transform.rotation.w = camera_to_base.transform.rotation.w

                self.br.sendTransform(obj_transform)

                self.get_logger().info(f'Successfully received data!')

            except tf2_ros.TransformException as e:
                self.get_logger().info(f'Could not transform base_link to child_link: {e}')
                return


            # Mark the detected fruit centers
            cv2.circle(rgb_image,(cX, cY), 5, (0, 255, 0), -1) 
            cv2.putText(rgb_image, "bad fruit", (cX - 40, cY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(rgb_image, (cX-30, cY-15), (cX+30, cY+15), (0,255,0), 3)

        cv2.imshow("fruits_tf_view", rgb_image)
        # cv2.imshow("Color_Image", self.cv_image)
        # cv2.imshow("Depth_Image", self.visualize_depth_image)
        # print(self.depth_image)
        cv2.imshow("cropped", self.cropped_rgb)
        cv2.waitKey(1)

        ros2_final = self.bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
        ros2_crop = self.bridge.cv2_to_imgmsg(self.cropped_rgb, encoding="bgr8")
        
        self.pub_crop.publish(ros2_crop)
        self.pub_final.publish(ros2_final)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()