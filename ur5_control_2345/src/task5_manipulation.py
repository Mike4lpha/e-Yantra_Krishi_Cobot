#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
import time
import signal
import numpy as np
from tf2_ros import TransformException
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from scipy.spatial.transform import Rotation
import tf2_ros
from std_srvs.srv import SetBool
from control_msgs.msg import JointJog

MAX_FRUIT_ID = 3
TEAM_ID = "2345"

class JointServoNode(Node):
    def __init__(self):

        super().__init__('ur5_manipulation_node')

        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.delivery_pub = self.create_publisher(
            Bool, '/fertilizer_delivered', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.pose_sub = self.create_subscription(Float64MultiArray, '/tcp_pose_raw', self.pose_callback, 10)
        self.dock_reached_sub = self.create_subscription(Bool, '/dock_reached', self.dock_reached_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Service clients
        self.magnet_client = self.create_client(SetBool, '/magnet')

        # Wait for services to be available
        self.wait_for_services()

        self.state = 0
        self.current_fruit_idx = 0

        # Timer-based loop (robust & non-blocking)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.tolerance = 0.1
        self.position_tolerance = 0.02
        self.rotation_tolerance = 0.05

        self.joint_recieved = False
        self.pose_received = False
        self.current_pose = None
        self.service_in_progress = False
        self.attached = False
        self.detached = True
        self.manipulation_enabled = False

        self.Kp1 = 1.0
        self.Kp2 = 1.0
        self.Kp3 = 1.0
        self.Kp4 = 1.0
        self.Kp5 = 1.0

    def wait_for_services(self):
        self.get_logger().info('Waiting for /magnet service...')
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /magnet...')
        self.get_logger().info('/magnet service is available.')

    def dock_reached_callback(self, msg):
        if msg.data and not self.manipulation_enabled:
            self.get_logger().info("Ebot Docked --> Starting Manipulation.")
            self.manipulation_enabled = True

    def joint_state_callback(self, msg):

        positions = msg.position
        velocities = msg.velocity

        self.base_pos = positions[0]
        self.upper_link_pos = positions[1]
        self.forearm_pos = positions[2]
        self.wrist_pos = positions[3]
        self.ee_pos = positions[4]

        self.joint_recieved = True

    def pose_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            self.get_logger().warn("tcp_pose_raw length < 6")
            return

        x, y, z = msg.data[0:3]
        rx, ry, rz = msg.data[3:6]

        # Convert rotation vector to quaternion
        quat = Rotation.from_rotvec([rx, ry, rz]).as_quat()

        self.current_pose = np.array([
            x, y, z,
            quat[0], quat[1], quat[2], quat[3]
        ], dtype=float)

        self.pose_received = True

    def reached(self, targets, indices):

        current = [
            self.base_pos,
            self.upper_link_pos,
            self.forearm_pos,
            self.wrist_pos,
            self.ee_pos
        ]

        for t, i in zip(targets, indices):
            if abs(t - current[i]) > self.tolerance:
                return False
        return True

    def go_to_fertilizer(self):

        base_target = -4.760179776174785
        upper_link_target = -1.5549295571044803
        forearm_target = -1.4469999229947546
        wrist_target = -0.09943140406421667
        ee_target = 1.6286361719219227

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def go_to_home(self):

        base_target = -3.1399936459290454
        upper_link_target = -0.5899780380094783
        forearm_target = -2.4900280481589636
        ee_target = 1.5699926358553273

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, 0.0, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, ee_target], [0,1,2,4])

    def bring_fertilizer(self):

        base_target = -4.8636828866571635
        upper_link_target = -1.2365404000245288
        forearm_target = -1.698317528814367
        wrist_target = -0.16744280290501556
        ee_target = 1.7329275735759788

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def mid_drop(self):
        Kp1 = 1.0
        Kp2 = 1.0
        Kp3 = 1.0
        Kp4 = 1.0
        Kp5 = 1.0

        base_target = -3.1673125542904987
        upper_link_target = -1.8706060691018391
        forearm_target = -1.38479771543943
        wrist_target = -1.4932392982238512
        ee_target = 1.5699926358553273

        speed1 = Kp1 * (base_target - self.base_pos)
        speed2 = Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = Kp3 * (forearm_target - self.forearm_pos)
        speed4 = Kp4 * (wrist_target - self.wrist_pos)
        speed5 = Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def go_to_drop_fertilizer(self):
        Kp1 = 1.0
        Kp2 = 1.0
        Kp3 = 1.0
        Kp4 = 1.0
        Kp5 = 1.0

        base_target = -3.014158422725833
        upper_link_target = - 2.6654966249490797
        forearm_target = -3.5840424977736623e-12
        wrist_target = -2.002100473449279
        ee_target = 1.5709835734017172

        speed1 = Kp1 * (base_target - self.base_pos)
        speed2 = Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = Kp3 * (forearm_target - self.forearm_pos)
        speed4 = Kp4 * (wrist_target - self.wrist_pos)
        speed5 = Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def go_to_fruits(self):

        base_target = -1.282366243740295
        upper_link_target = -1.6406106404957568
        forearm_target = -1.5106606642436178
        wrist_target = -1.4932392982238512
        ee_target = 1.5699926358553273

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def fruit1(self):

        # base_target = -1.3165195677435557
        # upper_link_target = -2.740853439425748
        # forearm_target = -0.9206929136990355
        # wrist_target = -1.0958317622785774
        # ee_target = 1.6120744236417563

        # base_target = -1.3163722902443806
        # upper_link_target = -2.596971190865977
        # forearm_target = -1.021221233986786
        # wrist_target = -1.0893185569528943
        # ee_target = 1.6122497264896383

        base_target = -1.2227392756729583 - 0.0525
        upper_link_target = -2.902666398748553 + 0.035 + 0.0175
        forearm_target = -0.6548186736020187
        wrist_target = -1.1458867356873128
        ee_target = 1.6118466586362727

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def above_fruit1(self):

        base_target = -1.3189346550917687
        upper_link_target = -2.2896020751795707
        forearm_target = -1.1846423497584856
        wrist_target = -1.2312580079910087
        ee_target = 1.6101460756412616

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def fruit2(self):

        # base_target = -1.2426589894676665
        # upper_link_target = -2.4850121589737797
        # forearm_target = -1.6350599076686683
        # wrist_target = -0.5848421283828217
        # ee_target = 1.6117798824796756

        # base_target = -1.242723351650584
        # upper_link_target = -2.292101935257892
        # forearm_target = -1.6429178882135067
        # wrist_target = -0.7694737713301253
        # ee_target = 1.6117780164651148

        base_target = -1.27691554852284 - 0.0175
        upper_link_target = -2.386468881046128 - 0.0875
        forearm_target = -1.4865748384611968
        wrist_target = -0.8239579160507691
        ee_target = 1.60959534311991

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def above_fruit2(self):

        base_target = -1.27691554852284
        upper_link_target = -2.266250649417549
        forearm_target = -1.6065402761900294
        wrist_target = -0.8319424971422845
        ee_target = 1.6120167086517032

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def fruit3(self):

        base_target = -1.1696172962485396 + 0.0175
        upper_link_target = -2.2154071298193077
        forearm_target = -2.020471128768955
        wrist_target = -0.4847572968946216
        ee_target = 1.6109964870946791

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def above_fruit3(self):

        base_target = -1.2187138982101022
        upper_link_target = -1.8589944270126677
        forearm_target = -2.010292985855276
        wrist_target = -0.8276236558591362
        ee_target = 1.6115865602116692

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def fruit4(self):

        # base_target = -1.1384509599826242
        # upper_link_target = -2.7525126166584215
        # forearm_target = -0.8666822552147588
        # wrist_target = -1.1002524571141024
        # ee_target = 1.6107349507791804

        base_target = -1.2227392756729583 + 0.0525 + 0.035
        upper_link_target = -2.902666398748553 + 0.035 + 0.0175
        forearm_target = -0.6548186736020187
        wrist_target = -1.1458867356873128
        ee_target = 1.6118466586362727

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def above_fruit4(self):

        base_target = -1.2819430647588679 + 0.0525 + 0.035
        upper_link_target = -2.753517037804284
        forearm_target = -0.37180513142051563
        wrist_target = -1.5775206731996227
        ee_target = 1.6121115012167477

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def fruit5(self):

        # base_target = -1.0553679147030832
        # upper_link_target = -2.5734042814644544
        # forearm_target = -1.47137275733177
        # wrist_target = -0.6530187449194466
        # ee_target = 1.609604670184639

        base_target = -1.0555503152186119 + 0.0175 - 0.012
        upper_link_target = -2.386468881046128 - 0.0875
        forearm_target = -1.4865748384611968
        wrist_target = -0.8239579160507691
        ee_target = 1.60959534311991

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def above_fruit5(self):

        base_target = -1.0593255245169306
        upper_link_target = -2.2096040195912616
        forearm_target = -1.4369028438316211
        wrist_target = -1.0460186800139442
        ee_target = 1.6066937219806126

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def fruit6(self):

        # base_target = -0.9310771064003496
        # upper_link_target = -2.2787593125699264
        # forearm_target = -1.9196880421834681
        # wrist_target = -0.5298087136861228
        # ee_target = 1.606106760436068

        base_target = -0.9719
        upper_link_target = -2.2018
        forearm_target = -1.8772
        wrist_target = -0.6272
        ee_target = 1.602

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def above_fruit6(self):

        base_target = -0.9222522084311503
        upper_link_target = -2.0380463085423166
        forearm_target = -1.931365466922592
        wrist_target = -0.7198995020338163
        ee_target = 1.607212148110116

        speed1 = self.Kp1 * (base_target - self.base_pos)
        speed2 = self.Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = self.Kp3 * (forearm_target - self.forearm_pos)
        speed4 = self.Kp4 * (wrist_target - self.wrist_pos)
        speed5 = self.Kp5 * (ee_target - self.ee_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, speed5, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target, upper_link_target, forearm_target, wrist_target, ee_target], [0,1,2,3,4])

    def go_above_drop(self):
        Kp1 = 1.0

        base_target = 0.11408851187744684

        speed1 = Kp1 * (base_target - self.base_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target], [0])

    def go_to_drop_fruits(self):
        Kp1 = 1.0
        Kp2 = 1.0
        Kp3 = 1.0
        Kp4 = 1.0

        base_target = 0.11408851187744684
        upper_link_target = -2.3593060910922596
        forearm_target = -0.7836198293373996
        wrist_target = -1.673865122135485

        speed1 = Kp1 * (base_target - self.base_pos)
        speed2 = Kp2 * (upper_link_target - self.upper_link_pos)
        speed3 = Kp3 * (forearm_target - self.forearm_pos)
        speed4 = Kp4 * (wrist_target - self.wrist_pos)

        cmd = JointJog()
        cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        cmd.velocities = [speed1, speed2, speed3, speed4, 0.0, 0.0]
        self.joint_pub.publish(cmd)

        return self.reached([base_target,upper_link_target, forearm_target, wrist_target], [0,1,2,3])

    def fetch_fertilizer_tf(self):

        # self.get_logger().info(" Checking for fertilizer...")
        fertilizer_pose = None
        source_frame = 'base_link'
        try:
            tf = self.tf_buffer.lookup_transform(
                source_frame, '2345_fertilizer_1', rclpy.time.Time()
            )
            t = tf.transform.translation
            r = tf.transform.rotation

            fertilizer_pose = [t.x, t.y + 0.02, t.z, r.x, r.y, r.z, r.w]

        except TransformException:
            self.get_logger().warn("fertilizer TF not available")

        return fertilizer_pose

    def fetch_all_fruit_tfs(self):
        fruit_poses = []
        above_fruit_poses = []
        source_frame = 'base_link'
        for fruit_id in range(1, MAX_FRUIT_ID + 1):
            target_frame = f'{TEAM_ID}_bad_fruit_{fruit_id}'
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    source_frame, target_frame, rclpy.time.Time()
                )
                t = transform_stamped.transform.translation
                r = transform_stamped.transform.rotation
                fruit_pose = [t.x, t.y, t.z+0.1, r.x, r.y, r.z, r.w]
                above_fruit_pose = [t.x, t.y, t.z+0.2, r.x, r.y, r.z, r.w]
                fruit_poses.append(fruit_pose)
                above_fruit_poses.append(above_fruit_pose)
                self.get_logger().info(
                    f"Found Fruit {fruit_id}: Pos({t.x:.2f}, {t.y:.2f}, {t.z:.2f})"
                )
            except TransformException:
                pass

        return fruit_poses, above_fruit_poses

    def compute_twist_error(self, current_pose, target_pose):
        pos_err = np.array(target_pose[:3]) - np.array(current_pose[:3])
        r_current = Rotation.from_quat(current_pose[3:])
        r_target = Rotation.from_quat(target_pose[3:])
        rot_error = (r_target * r_current.inv()).as_rotvec()

        pos_err_norm = np.linalg.norm(pos_err)
        rot_err_norm = np.linalg.norm(rot_error)

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

        return twist, pos_err_norm, rot_err_norm

    def control_loop(self):
        if not self.manipulation_enabled:
            return

        if not self.joint_recieved:
            self.get_logger().info("Joint not recieved")
            return

        if not self.pose_received:
            self.get_logger().info("Pose not recieved")
            return

        if self.service_in_progress:
            self.get_logger().info("Service in progress")
            self.stop_twist()
            self.stop_joint()
            return

        # if self.state == -1:
        #     self.fertilizer_pose = self.fetch_fertilizer_tf()
        #     self.state = 0
        #     self.get_logger().info(f"state: {self.state}")

        # STATE 0 — GO_FERTILIZER
        if self.state == 0:
            self.get_logger().info("Initializing arm...")
            if self.bring_fertilizer():
                self.state = 1
                self.get_logger().info(f"state: {self.state}")

        # STATE 1 — FETCH FERTILIZER
        elif self.state == 1:
            if self.go_to_fertilizer():
                if not self.attached and self.detached:
                    self.call_magnet_service(True)
                    self.get_logger().info("Activating magnet...")
                elif self.attached:
                    self.state = 2
                    self.get_logger().info(f"state: {self.state}")


        # STATE 2
        elif self.state == 2:
            if self.bring_fertilizer():
                self.state = 3
                self.get_logger().info(f"state: {self.state}")

        # STATE 3
        elif self.state == 3:
            if self.go_to_home():
                self.state = 4
                self.get_logger().info(f"state: {self.state}")

        elif self.state == 4:
            if self.mid_drop():
                self.state = 5
                self.get_logger().info(f"state: {self.state}")

        # STATE 4 — GO_DROP_FERTILIZER
        elif self.state == 5:
            if self.go_to_drop_fertilizer():
                if self.attached and not self.detached:
                    self.call_magnet_service(False)
                    self.get_logger().info("Activating magnet...")
                elif self.detached:
                    delivered_msg = Bool()
                    delivered_msg.data = True
                    self.delivery_pub.publish(delivered_msg)
                    self.state = 6
                    self.get_logger().info(f"state: {self.state}")

        # STATE 5 — FETCH_FRUITS
        elif self.state == 6:
            self.fruit_poses = self.fetch_all_fruit_tfs()
            self.get_logger().info("Recieved")
            self.state = 7
            self.get_logger().info(f"state: {self.state}")

        # STATE 6 — GO TO FRUIT
        elif self.state == 7:
            if self.go_to_fruits():
                self.state = 8
                self.get_logger().info(f"state: {self.state}")

        # STATE 7 — PICK FRUIT
        elif self.state == 8:
            if self.fruit3():
                if not self.attached and self.detached:
                    self.call_magnet_service(True)
                    self.get_logger().info("Activating magnet...")
                elif self.attached:
                    self.state = 9  # move next only when magnet is attached
                    self.get_logger().info(f"state: {self.state}")

        elif self.state == 9:
            if self.above_fruit3():
                self.state = 10
                self.get_logger().info(f"state: {self.state}")

        # STATE 10 — DROP_FRUIT
        elif self.state == 10:
            if self.go_to_drop_fruits():
                if self.attached and not self.detached:
                    self.call_magnet_service(False)
                    self.get_logger().info("Activating magnet...")
                elif self.detached:
                    self.state = 11  # move next only when magnet is attached
                    self.get_logger().info(f"state: {self.state}")

        elif self.state == 11:
            if self.go_to_fruits():
                self.state = 12
                self.get_logger().info(f"state: {self.state}")

        elif self.state == 12:
            if self.fruit1():
                if not self.attached and self.detached:
                    self.call_magnet_service(True)
                    self.get_logger().info("Activating magnet...")
                elif self.attached:
                    self.state = 13
                    self.get_logger().info(f"state: {self.state}")

        elif self.state == 13:
            if self.above_fruit1():
                self.state = 14
                self.get_logger().info(f"state: {self.state}")

        # STATE 10 — DROP_FRUIT
        elif self.state == 14:
            if self.go_to_drop_fruits():
                if self.attached and not self.detached:
                    self.call_magnet_service(False)
                    self.get_logger().info("Activating magnet...")
                elif self.detached:
                    self.state = 15
                    self.get_logger().info(f"state: {self.state}")

        elif self.state == 15:
            if self.go_to_fruits():
                self.state = 16
                self.get_logger().info(f"state: {self.state}")

        elif self.state == 16:
            if self.fruit5():
                if not self.attached and self.detached:
                    self.call_magnet_service(True)
                    self.get_logger().info("Activating magnet...")
                elif self.attached:
                    self.state = 17
                    self.get_logger().info(f"state: {self.state}")

        elif self.state == 17:
            if self.above_fruit5():
                self.state = 18
                self.get_logger().info(f"state: {self.state}")

        # STATE 10 — DROP_FRUIT
        elif self.state == 18:
            if self.go_to_drop_fruits():
                if self.attached and not self.detached:
                    self.call_magnet_service(False)
                    self.get_logger().info("Activating magnet...")
                elif self.detached:
                    self.state = 19
                    self.get_logger().info(f"state: {self.state}")

        elif self.state == 19:
            if self.go_to_fruits():
                self.state = 20
                self.get_logger().info(f"state: {self.state}")

        # STATE 23 — DONE
        elif self.state == 20:
            self.get_logger().info("✅ Task completed")
            self.control_timer.cancel()

    def call_magnet_service(self, value: bool):
        """Call /magnet SetBool service with data=value (True=on/attach, False=off/detach)"""

        if self.service_in_progress:
            return

        if not self.magnet_client.service_is_ready():
            self.get_logger().warn("Magnet service not ready.")
            return

        req = SetBool.Request()
        req.data = bool(value)
        self.service_in_progress = True
        self.get_logger().info("Service in progress..")
        future = self.magnet_client.call_async(req)
        future.add_done_callback(lambda f: self.magnet_response_callback(f, value))

    def magnet_response_callback(self, future, requested_value):
        try:
            result = future.result()
            if result is not None and result.success:
                if requested_value:
                    self.get_logger().info("Magnet ON succeeded.")
                    self.attached = True
                    self.detached = False
                    self.service_in_progress = False
                else:
                    self.get_logger().info("Magnet OFF succeeded.")
                    self.detached = True
                    self.attached = False
                    self.service_in_progress = False
            else:
                self.get_logger().error(f"Magnet service reported failure: {result}")
                self.service_in_progress = False
        except Exception as e:
            self.get_logger().error(f"Magnet service call exception: {e}")
            self.service_in_progress = False

    def stop_twist(self):
        ts = TwistStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        self.twist_pub.publish(ts)

    def stop_joint(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        self.joint_pub.publish(js)

def main(args=None):
    rclpy.init(args=args)
    node = JointServoNode()

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
            node.stop_twist()
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
