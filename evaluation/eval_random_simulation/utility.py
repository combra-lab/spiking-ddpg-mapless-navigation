import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import math
from shapely.geometry import Point, Polygon
import sys
sys.path.append('../../')
from training.train_ddpg.ddpg_networks import ActorNet
from training.train_spiking_ddpg.sddpg_networks import ActorNetSpiking


def load_test_actor_network(dir, state_num=22, action_num=2, dim=(256, 256, 256)):
    """
    Load actor network for testing
    :param dir: directory of pt file
    :return: actor_net
    """
    actor_net = ActorNet(state_num, action_num,
                         hidden1=dim[0],
                         hidden2=dim[1],
                         hidden3=dim[2])
    actor_net.load_state_dict(torch.load(dir, map_location=lambda storage, loc: storage))
    return actor_net


def load_test_actor_snn_network(weight_dir, bias_dir, device, batch_window=50,
                                state_num=24, action_num=2, dim=(256, 256, 256)):
    """
    Load actor snn network for testing
    :param weight_dir: directory of numpy weights
    :param bias_dir: directory of numpy bias
    :param state_num: number of states
    :param action_num: number of actions
    :param dim: net dim
    :return: actor_net
    """
    actor_net = ActorNetSpiking(state_num, action_num, device,
                                batch_window=batch_window,
                                hidden1=dim[0],
                                hidden2=dim[1],
                                hidden3=dim[2])
    weights = pickle.load(open(weight_dir, 'rb'))
    bias = pickle.load(open(bias_dir, 'rb'))
    actor_net.fc1.weight = nn.Parameter(torch.from_numpy(weights[0]))
    actor_net.fc2.weight = nn.Parameter(torch.from_numpy(weights[1]))
    actor_net.fc3.weight = nn.Parameter(torch.from_numpy(weights[2]))
    actor_net.fc4.weight = nn.Parameter(torch.from_numpy(weights[3]))
    actor_net.fc1.bias = nn.Parameter(torch.from_numpy(bias[0]))
    actor_net.fc2.bias = nn.Parameter(torch.from_numpy(bias[1]))
    actor_net.fc3.bias = nn.Parameter(torch.from_numpy(bias[2]))
    actor_net.fc4.bias = nn.Parameter(torch.from_numpy(bias[3]))
    return actor_net


def gen_goal_position_list(poly_list, env_size=((-6, 6), (-6, 6)), obs_near_th=0.5, sample_step=0.1):
    """
    Generate list of goal positions
    :param poly_list: list of obstacle polygon
    :param env_size: size of the environment
    :param obs_near_th: Threshold for near an obstacle
    :param sample_step: sample step for goal generation
    :return: goal position list
    """
    goal_pos_list = []
    x_pos, y_pos = np.mgrid[env_size[0][0]:env_size[0][1]:sample_step, env_size[1][0]:env_size[1][1]:sample_step]
    for x in range(x_pos.shape[0]):
        for y in range(x_pos.shape[1]):
            tmp_pos = [x_pos[x, y], y_pos[x, y]]
            tmp_point = Point(tmp_pos[0], tmp_pos[1])
            near_obstacle = False
            for poly in poly_list:
                tmp_dis = tmp_point.distance(poly)
                if tmp_dis < obs_near_th:
                    near_obstacle = True
            if near_obstacle is False:
                goal_pos_list.append(tmp_pos)
    return goal_pos_list


def gen_polygon_exterior_list(poly_point_list):
    """
    Generate list of obstacle in the environment as polygon exterior list
    :param poly_point_list: list of points of polygon (with first always be the out wall)
    :return: polygon exterior list
    """
    poly_list = []
    for i, points in enumerate(poly_point_list, 0):
        tmp_poly = Polygon(points)
        if i > 0:
            poly_list.append(tmp_poly)
        else:
            poly_list.append(tmp_poly.exterior)
    return poly_list


def gen_test_env_poly_list_env():
    """
    Generate Poly list of test environment
    :return: poly_list
    """
    env = [(-10, 10), (10, 10), (10, -10), (-10, -10)]
    obs1 = [(-10, -5), (-6, -5), (-6, -5.5), (-10, -5.5)]
    obs2 = [(-5.5, -6), (-5, -6), (-5, -10), (-5.5, -10)]
    obs3 = [(-3, -5), (4, -5), (4, -6), (-3, -6)]
    obs4 = [(6, -6), (10, -6), (10, -7), (6, -7)]
    obs5 = [(-6, -2), (-5, -2), (-5, -3), (-6, -3)]
    obs6 = [(-9.25, 1), (-4, 1), (-4, 0), (-9.25, 0)]
    obs7 = [(-7, 5), (-5, 5), (-5, 3), (-7, 3)]
    obs8 = [(-7, 7), (-3, 7), (-3, 5), (-7, 5)]
    obs9 = [(0, 7), (1, 5), (-1, 5)]
    obs10 = [(4, 6), (5, 7), (6, 6), (5, 5)]
    obs11 = [(4.5, 7.5), (5.5, 8.5), (6.5, 7.5), (5.5, 6.5)]
    obs12 = [(6.5, 8), (8, 8), (8, 6.5), (6.5, 6.5)]
    obs13 = [(6, 6), (7, 6), (7, 5), (6, 5)]
    obs14 = [(5, 4), (5.5, 4), (5.5, 2.75), (5, 2.75)]
    obs15 = [(5.5, 4), (9, 4), (9, 3.5), (5.5, 3.5)]
    obs16 = [(5, 1.5), (5.5, 1.5), (5.5, -1.5), (5, -1.5)]
    obs17 = [(5.5, 0.5), (8, 0.5), (8, -0.5), (5.5, -0.5)]
    obs18 = [(5, -2.75), (5.5, -2.75), (5.5, -4), (5, -4)]
    obs19 = [(5.5, -3.5), (9, -3.5), (9, -4), (5.5, -4)]
    obs20 = [(-3, 3), (1.5, 3), (0, 1), (-3, 1)]
    obs21 = [(1, 0), (3, 2), (3, -2)]
    obs22 = [(-3, -3), (1.5, -3), (0, -1), (-3, -1)]
    obs23 = [(-1, -8), (1, -8), (0, -10)]
    obs24 = [(-8, -7), (-7, -7), (-7, -8), (-8, -8)]
    obs25 = [(-8, -1.75), (-6, -1.75), (-6, -2.75), (-8, -2.75)]
    obs26 = [(3, -8), (4, -7), (4.5, -7.5), (3.5, -8.5)]
    obs27 = [(-5, -4), (-4, -4), (-4, -5), (-5, -5)]
    obs28 = [(-2, 9.5), (-1.5, 10), (-0.5, 9), (-1, 8.5)]
    obs29 = [(-10, 8), (-9, 8), (-9, 6), (-10, 6)]
    obs30 = [(-6, 10), (-4, 10), (-4, 9), (-6, 9)]
    obs31 = [(1, 9), (3, 9), (2, 7)]
    obs32 = [(3.5, 4), (4, 3.5), (3.5, 3), (3, 3.5)]
    obs33 = [(3.2, -3), (4, -3.2), (3.8, -4), (3, -3.8)]
    poly_raw_list = [env, obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9,
                     obs10, obs11, obs12, obs13, obs14, obs15, obs16, obs17, obs18,
                     obs19, obs20, obs21, obs22, obs23, obs24, obs25, obs26, obs27,
                     obs28, obs29, obs30, obs31, obs32, obs33]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    return poly_list, poly_raw_list


def gen_robot_pose_n_goal_list(size, poly_list, env_size=((-10, 10), (-10, 10)),
                               goal_th=1.0, robot_goal_diff=3.0):
    """
    Randomly generate robot pose and goal
    :param size: number of the list
    :return: pose_list, goal_list
    """
    all_position_list = gen_goal_position_list(poly_list, env_size=env_size, obs_near_th=goal_th)
    pose_list = []
    goal_list = []
    for num in range(size):
        goal = random.choice(all_position_list)
        pose = random.choice(all_position_list)
        distance = math.sqrt((goal[0] - pose[0])**2 + (goal[1] - pose[1])**2)
        while distance < robot_goal_diff:
            pose = random.choice(all_position_list)
            distance = math.sqrt((goal[0] - pose[0])**2 + (goal[1] - pose[1])**2)
        pose.append(random.random()*2*math.pi)
        pose_list.append(pose)
        goal_list.append(goal)
    return pose_list, goal_list

