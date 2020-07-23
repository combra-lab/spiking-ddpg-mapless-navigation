import rospy
import math
import pickle
import copy
import time
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Odometry
from simple_laserscan.msg import SimpleScan
import sys
sys.path.append('../../')
from training.train_ddpg.ddpg_agent import Agent
from training.utility import *


class RosNode:
    """ ROS Node For ROS-Loihi Interaction """
    def __init__(self,
                 agent,
                 goal_list,
                 run_time,
                 laser_scan_half_num=9,
                 laser_scan_min_dis=0.35,
                 ros_rate=10,
                 goal_th=0.5,
                 goal_dis_min_dis=0.3,
                 is_pos_neg=False,
                 log_print=False):
        """

        :param agent: Actor agent
        :param goal_list: list of goal positions
        :param run_time: running time of simulation in seconds
        :param laser_scan_half_num: half number of scan points
        :param laser_scan_min_dis: Min laser scan distance
        :param ros_rate: ros update rate
        :param goal_th: Threshold distance to reset goal
        :param log_print: if or not print log message
        """
        self.agent = agent
        self.goal_list = goal_list
        self.run_steps = run_time * ros_rate
        self.laser_scan_half_num = laser_scan_half_num
        self.laser_scan_min_dis = laser_scan_min_dis
        self.ros_rate = ros_rate
        self.goal_th = goal_th
        self.goal_dis_min_dis = goal_dis_min_dis
        self.is_pos_neg = is_pos_neg
        self.log_print = log_print
        # Robot State
        self.robot_pose_init = False
        self.robot_spd_init = False
        self.robot_pose = [0., 0., 0.]
        self.robot_spd = [0., 0.]
        self.robot_scan_init = False
        self.robot_scan = np.zeros(2 * laser_scan_half_num)
        # Init ROS Node
        rospy.init_node("ros_ddpg_inter")
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self._robot_pose_cb)
        rospy.Subscriber('odom', Odometry, self._robot_spd_cb)
        rospy.Subscriber('simplescan', SimpleScan, self._robot_scan_cb)
        self.pub_action = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=5)
        self.pub_goal = rospy.Publisher('test_goal_pos', PointStamped, queue_size=5)
        # Init Subscriber
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
            tmp_state = [goal_dir, tmp_goal_dis, tmp_robot_state[1][0], tmp_robot_state[1][1]]
            tmp_state.extend(tmp_robot_scan.tolist())
            if self.is_pos_neg:
                rescale_state = ddpg_state_2_spike_value_state(tmp_state, 24)
            else:
                rescale_state = ddpg_state_rescale(tmp_state, 22)
            raw_action = self.agent.act(rescale_state, explore=True, train=False)
            action = wheeled_network_2_robot_action_decoder(
                raw_action, 0.5, 0.05
            )
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
    WEIGHT_FILE = '../saved_model/ddpg_poisson.pt'
    GOAL_LIST = [[7.3, 2.5], [7.7, -3], [5.3, -5], [0, -5.2], [2.0, -9.2], [11.4, -9.2],
                 [13, -6], [11, -4.5], [13.5, -10], [10, -13.5], [7, -17], [1, -17], [0.5, -15.2], [7, -13.2], [0, -12.7]]
    USE_CUDA = True
    IS_POS_NEG = True
    IS_POISSON = True
    STATE_NUM = 18 + 4
    if IS_POS_NEG:
        RESCALE_STATE_NUM = STATE_NUM + 2
    else:
        RESCALE_STATE_NUM = STATE_NUM
    ACTION_NUM = 2
    POISSON_WIN = 50
    RAND_END = 0.01
    agent = Agent(STATE_NUM, ACTION_NUM, RESCALE_STATE_NUM, poisson_window=POISSON_WIN, use_poisson=IS_POISSON,
                  epsilon_end=RAND_END, use_cuda=USE_CUDA)
    agent.load(WEIGHT_FILE)
    sim_time = 400
    ros_node = RosNode(agent, GOAL_LIST, sim_time, is_pos_neg=IS_POS_NEG)
    record_pos, record_time, record_dis = ros_node.run_ros()
