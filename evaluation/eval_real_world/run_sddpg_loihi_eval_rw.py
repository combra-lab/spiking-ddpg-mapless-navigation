import rospy
import math
import copy
import time
import pickle
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Odometry
from simple_laserscan.msg import SimpleScan
import sys
sys.path.append('../../')
from evaluation.loihi_network.utility import robot_2_goal_dis_dir, \
    robot_state_2_loihi_spike_prob, loihi_spikes_2_robot_action, combine_multiple_into_one_int
from evaluation.loihi_network.spiking_actor_net_loihi import SpikingActorNet


class RosNode:
    """ ROS Node For ROS-Loihi Interaction """
    def __init__(self,
                 goal_list,
                 encoder_channel,
                 decoder_channel,
                 run_time,
                 window=50,
                 laser_scan_half_num=9,
                 laser_scan_min_dis=0.35,
                 goal_dis_min_dis=0.3,
                 ros_rate=10,
                 goal_th=0.5,
                 log_print=False):
        """

        :param goal_list: list of goal positions
        :param encoder_channel: Loihi encoder channel
        :param decoder_channel: Loihi decoder channel
        :param run_time: running time of simulation in seconds
        :param laser_scan_half_num: half number of scan points
        :param laser_scan_min_dis: Min laser scan distance
        :param goal_dis_min_dis:
        :param ros_rate: ros update rate
        :param goal_th: Threshold distance to reset goal
        :param log_print: if or not print log message
        """
        self.goal_list = goal_list
        self.encoder_channel = encoder_channel
        self.decoder_channel = decoder_channel
        self.run_steps = run_time * ros_rate
        self.window = window
        self.laser_scan_half_num = laser_scan_half_num
        self.laser_scan_min_dis = laser_scan_min_dis
        self.goal_dis_min_dis = goal_dis_min_dis
        self.ros_rate = ros_rate
        self.goal_th = goal_th
        self.log_print = log_print
        # Robot State
        self.robot_pose_init = False
        self.robot_spd_init = False
        self.robot_pose = [0., 0., 0.]
        self.robot_spd = [0., 0.]
        self.robot_scan_init = False
        self.robot_scan = np.zeros(2 * laser_scan_half_num)
        # Init ROS Node
        rospy.init_node("ros_loihi_inter")
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self._robot_pose_cb)
        rospy.Subscriber('odom', Odometry, self._robot_spd_cb)
        rospy.Subscriber('simplescan', SimpleScan, self._robot_scan_cb)
        self.pub_action = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=5)
        self.pub_goal = rospy.Publisher('test_goal_pos', PointStamped, queue_size=5)
        # Init Subscriber
        # while not self.robot_pose_init:
        #     continue
        while not self.robot_spd_init:
            continue
        while not self.robot_scan_init:
            continue
        rospy.loginfo("Finish Subcriber Init ...")

    def run_ros(self):
        """
        Run ROS Node
        :return:
        """
        run_distance = 0
        run_time = 0
        robot_pose = {"x": [], "y": [], "theta": []}
        rate = rospy.Rate(self.ros_rate)
        ita = 0
        goal_ita = 0
        new_target_msg = PointStamped()
        new_target_msg.header.frame_id = 'map'
        new_target_msg.point.x = self.goal_list[goal_ita][0]
        new_target_msg.point.y = self.goal_list[goal_ita][1]
        self.pub_goal.publish(new_target_msg)
        start_time = time.time()
        while not rospy.is_shutdown():
            tmp_robot_state = [copy.deepcopy(self.robot_pose), copy.deepcopy(self.robot_spd)]
            tmp_robot_scan = copy.deepcopy(self.robot_scan)
            tmp_robot_scan = self.laser_scan_min_dis / tmp_robot_scan
            tmp_robot_scan = np.clip(tmp_robot_scan, 0, 1)
            goal_dis, goal_dir = robot_2_goal_dis_dir(tmp_robot_state[0], self.goal_list[goal_ita])
            if goal_dis < self.goal_th:
                goal_ita += 1
                if goal_ita == len(self.goal_list):
                    break
                new_target_msg = PointStamped()
                new_target_msg.header.frame_id = 'map'
                new_target_msg.point.x = self.goal_list[goal_ita][0]
                new_target_msg.point.y = self.goal_list[goal_ita][1]
                goal_dis, goal_dir = robot_2_goal_dis_dir(tmp_robot_state[0], self.goal_list[goal_ita])
            self.pub_goal.publish(new_target_msg)
            tmp_goal_dis = goal_dis
            if tmp_goal_dis == 0:
                tmp_goal_dis = 1
            else:
                tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis
                if tmp_goal_dis > 1:
                    tmp_goal_dis = 1
            state_spike_prob = robot_state_2_loihi_spike_prob(
                [goal_dir, tmp_goal_dis, tmp_robot_state[1][0], tmp_robot_state[1][1]], tmp_robot_scan
            )
            encode_spike_prob = combine_multiple_into_one_int(state_spike_prob)
            self.encoder_channel.write(len(encode_spike_prob), encode_spike_prob)
            out_spikes = self.decoder_channel.read(2)
            action = loihi_spikes_2_robot_action(out_spikes, window=self.window)
            move_cmd = Twist()
            move_cmd.linear.x = action[0]
            move_cmd.angular.z = action[1]
            self.pub_action.publish(move_cmd)
            robot_pose["x"].append(tmp_robot_state[0][0])
            robot_pose["y"].append(tmp_robot_state[0][1])
            robot_pose["theta"].append(tmp_robot_state[0][2])
            if len(robot_pose["x"]) > 1:
                tmp_delta_dis = math.sqrt((robot_pose["x"][-1] - robot_pose["x"][-2])**2 +
                                          (robot_pose["y"][-1] - robot_pose["y"][-2])**2)
                run_distance += tmp_delta_dis
            ita += 1
            if ita == self.run_steps:
                break
            rate.sleep()
        end_time = time.time()
        run_time = end_time - start_time
        return robot_pose, run_time, run_distance

    def _robot_pose_cb(self, msg):
        """
        Callback function for robot pose
        :param msg: message
        """
        if self.robot_pose_init is False:
            self.robot_pose_init = True
        quat = [msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w]
        siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
        cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def _robot_spd_cb(self, msg):
        """
        Callback function for robot speed
        :param msg: message
        """
        if self.robot_spd_init is False:
            self.robot_spd_init = True
        self.robot_spd = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

    def _robot_scan_cb(self, msg):
        """
        Callback function for robot laser scan
        :param msg: message
        """
        if self.robot_scan_init is False:
            self.robot_scan_init = True
        tmp_robot_scan_ita = 0
        for num in range(self.laser_scan_half_num):
            ita = self.laser_scan_half_num - num - 1
            self.robot_scan[tmp_robot_scan_ita] = msg.data[ita]
            tmp_robot_scan_ita += 1
        for num in range(self.laser_scan_half_num):
            ita = len(msg.data) - num - 1
            self.robot_scan[tmp_robot_scan_ita] = msg.data[ita]
            tmp_robot_scan_ita += 1


if __name__ == "__main__":
    weights_dir = '../saved_model/sddpg_bw_5_weights.p'
    bias_dir = '../saved_model/sddpg_bw_5_bias.p'
    GOAL_LIST = [[7.3, 2.5], [7.7, -3], [5.3, -5], [0, -5.2], [2.0, -9.2], [11.4, -9.2],
                 [13, -6], [11, -4.5], [13.5, -10], [10, -13.5], [7, -17], [1, -17], [0.5, -15.2], [6.75, -13.2], [0, -12.7]]
    raw_weights = pickle.load(open(weights_dir, 'rb'))
    raw_bias = pickle.load(open(bias_dir, 'rb'))
    core_list_input = [0 for _ in range(24)]
    core_list_hidden1 = [num // 128 + 3 for num in range(256)]
    core_list_hidden2 = [num // 128 + 5 for num in range(256)]
    core_list_hidden3 = [num // 128 + 7 for num in range(256)]
    core_list_out = [2, 2]
    core_list_bias = [1, 1, 1, 1]
    core_list = [core_list_input, core_list_hidden1, core_list_hidden2, core_list_hidden3, core_list_out, core_list_bias]
    snn = SpikingActorNet(raw_weights, raw_bias, core_list)
    board, in_channel, out_channel = snn.setup_snn(print_axon=True)
    sim_time = 400
    sim_window = 5
    loihi_window = sim_window + 5
    loihi_steps = sim_time * loihi_window * 10
    board.startDriver()
    board.run(loihi_steps, aSync=True)
    ros_node = RosNode(GOAL_LIST, in_channel, out_channel, sim_time, window=sim_window)
    record_pos, record_time, record_dis = ros_node.run_ros()
    board.finishRun()
    board.disconnect()

