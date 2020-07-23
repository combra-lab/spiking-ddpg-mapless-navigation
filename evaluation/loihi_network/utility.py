import numpy as np
import copy
import math


def pytorch_trained_snn_param_2_loihi_snn_param(weight, bias, vth):
    """
    Transform pytorch weights, bias, and vth based on the scale
    :param weight: pytorch weights
    :param bias: pytorch bias
    :param vth: pytorch vth
    :return: weights_dict, new_vth, scale_factor
    """
    max_w = np.amax(weight)
    min_w = np.amin(weight)
    max_b = np.amax(bias)
    min_b = np.amin(bias)
    '''
    First find the scale factor

    Method:
        1. Find max absolute value between [max_w, min_w, max_b, min_b]
        2. Then the scale factor is 255 divide the max absolute value
    '''
    max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
    scale_factor = 255. / max_abs_value
    '''
    Second Compute New Loihi Voltage Threshold
    '''
    new_vth = int(vth * scale_factor)
    '''
    Third Compute Scaled Loihi Weight and Bias
    '''
    new_w = np.clip(weight * scale_factor, -255, 255)
    new_w = new_w.astype(int)
    new_b = np.clip(bias * scale_factor, -255, 255)
    new_b = new_b.astype(int)
    pos_w = copy.deepcopy(new_w)
    pos_w[new_w < 0] = 0
    pos_w_mask = np.int_(new_w > 0)
    neg_w = copy.deepcopy(new_w)
    neg_w[new_w > 0] = 0
    neg_w_mask = np.int_(new_w < 0)
    pos_b = copy.deepcopy(new_b)
    pos_b[new_b < 0] = 0
    pos_b_mask = np.int_(new_b > 0)
    pos_b = pos_b.reshape((-1, 1))
    pos_b_mask = pos_b_mask.reshape((-1, 1))
    neg_b = copy.deepcopy(new_b)
    neg_b[new_b > 0] = 0
    neg_b_mask = np.int_(new_b < 0)
    neg_b = neg_b.reshape((-1, 1))
    neg_b_mask = neg_b_mask.reshape((-1, 1))
    '''
    Generate weights_dict and return
    '''
    weights_dict = {'pos_w': pos_w,
                    'neg_w': neg_w,
                    'pos_w_mask': pos_w_mask,
                    'neg_w_mask': neg_w_mask,
                    'pos_b': pos_b,
                    'neg_b': neg_b,
                    'pos_b_mask': pos_b_mask,
                    'neg_b_mask': neg_b_mask}
    return weights_dict, new_vth, scale_factor


def pytorch_trained_snn_param_2_loihi_snn_param_amp(weight, bias, vth, bias_amp):
    """
    Transform pytorch weights, bias, and vth based on the scale
    :param weight: pytorch weights
    :param bias: pytorch bias
    :param vth: pytorch vth
    :param bias_amp: number of bias neurons
    :return: weights_dict, new_vth, scale_factor
    """
    bias = bias / bias_amp
    max_w = np.amax(weight)
    min_w = np.amin(weight)
    max_b = np.amax(bias)
    min_b = np.amin(bias)
    print(max_w, min_w, max_b, min_b)
    '''
    First find the scale factor

    Method:
        1. Find max absolute value between [max_w, min_w, max_b, min_b]
        2. Then the scale factor is 255 divide the max absolute value
    '''
    max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
    scale_factor = 255. / max_abs_value
    '''
    Second Compute New Loihi Voltage Threshold
    '''
    new_vth = int(vth * scale_factor)
    '''
    Third Compute Scaled Loihi Weight and Bias
    '''
    new_w = np.clip(weight * scale_factor, -255, 255)
    new_w = new_w.astype(int)
    new_b = np.clip(bias * scale_factor, -255, 255)
    new_b = new_b.astype(int)
    pos_w = copy.deepcopy(new_w)
    pos_w[new_w < 0] = 0
    pos_w_mask = np.int_(new_w > 0)
    neg_w = copy.deepcopy(new_w)
    neg_w[new_w > 0] = 0
    neg_w_mask = np.int_(new_w < 0)
    pos_b = copy.deepcopy(new_b)
    pos_b[new_b < 0] = 0
    pos_b_mask = np.int_(new_b > 0)
    pos_b = pos_b.reshape((-1, 1))
    pos_b_mask = pos_b_mask.reshape((-1, 1))
    neg_b = copy.deepcopy(new_b)
    neg_b[new_b > 0] = 0
    neg_b_mask = np.int_(new_b < 0)
    neg_b = neg_b.reshape((-1, 1))
    neg_b_mask = neg_b_mask.reshape((-1, 1))
    '''
    Multiple bias neurons
    '''
    if bias_amp > 1:
        amp_pos_b = np.zeros((pos_b.shape[0], bias_amp))
        amp_pos_b_mask = np.int_(np.zeros((pos_b.shape[0], bias_amp)))
        amp_neg_b = np.zeros((neg_b.shape[0], bias_amp))
        amp_neg_b_mask = np.int_(np.zeros((neg_b.shape[0], bias_amp)))
        for amp in range(bias_amp):
            amp_pos_b[:, amp] = pos_b[:, 0]
            amp_pos_b_mask[:, amp] = amp_pos_b_mask[:, 0]
            amp_neg_b[:, amp] = amp_neg_b[:, 0]
            amp_neg_b_mask[:, amp] = amp_neg_b_mask[:, 0]
        pos_b = amp_pos_b
        pos_b_mask = amp_pos_b_mask
        neg_b = amp_neg_b
        neg_b_mask = amp_neg_b_mask
    '''
    Generate weights_dict and return
    '''
    weights_dict = {'pos_w': pos_w,
                    'neg_w': neg_w,
                    'pos_w_mask': pos_w_mask,
                    'neg_w_mask': neg_w_mask,
                    'pos_b': pos_b,
                    'neg_b': neg_b,
                    'pos_b_mask': pos_b_mask,
                    'neg_b_mask': neg_b_mask}
    return weights_dict, new_vth, scale_factor


def robot_2_goal_dis_dir(robot_pose, goal_pos):
    """
    Compuate Relative Distance and Direction between robot and goal
    :param robot_pose: robot pose
    :param goal_pos: goal position
    :return: distance, direction
    """
    delta_x = goal_pos[0] - robot_pose[0]
    delta_y = goal_pos[1] - robot_pose[1]
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    ego_direction = math.atan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    while robot_direction < 0:
        robot_direction += 2 * math.pi
    while robot_direction > 2 * math.pi:
        robot_direction -= 2 * math.pi
    while ego_direction < 0:
        ego_direction += 2 * math.pi
    while ego_direction > 2 * math.pi:
        ego_direction -= 2 * math.pi
    pos_dir = abs(ego_direction - robot_direction)
    neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
    if pos_dir <= neg_dir:
        direction = math.copysign(pos_dir, ego_direction - robot_direction)
    else:
        direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
    return distance, direction


def robot_state_2_loihi_spike_gaps(robot_state,
                                   robot_scan,
                                   goal_dir_range=math.pi,
                                   linear_spd_range=0.5,
                                   angular_spd_range=2.0):
    """
    Transform Robot State (Relative Goal Direction, Linear Spd, Angular Spd) to Loihi Spike Time Gaps
    :param robot_state: robot state
    :param robot_scan: robot scan
    :param goal_dir_range: goal dir range
    :param linear_spd_range: linear spd range
    :param angular_spd_range: angular spd range
    :param max_gap: max time gap
    :return: loihi_spike_gaps
    """
    loihi_spike_gaps = [0 for _ in range(6 + len(robot_scan))]
    '''
    Goal Relative Direction
    '''
    if robot_state[0] >= 0:
        spike_rate = min(robot_state[0] / goal_dir_range, 1)
        spike_rate = max(spike_rate, 0.0001)
        spike_gap = round((1 / spike_rate) * 100)
        loihi_spike_gaps[0] = spike_gap
    else:
        spike_rate = min(abs(robot_state[0]) / goal_dir_range, 1)
        spike_rate = max(spike_rate, 0.0001)
        spike_gap = round((1 / spike_rate) * 100)
        loihi_spike_gaps[1] = spike_gap
    '''
    Goal Distance
    '''
    spike_rate = max(robot_state[1], 0.0001)
    spike_gap = round((1 / spike_rate) * 100)
    loihi_spike_gaps[2] = spike_gap
    '''
    Linear Spd
    '''
    spike_rate = min(abs(robot_state[2]) / linear_spd_range, 1)
    spike_rate = max(spike_rate, 0.0001)
    spike_gap = round((1 / spike_rate) * 100)
    loihi_spike_gaps[3] = spike_gap
    '''
    Angular Spd
    '''
    if robot_state[3] >= 0:
        spike_rate = min(robot_state[3] / angular_spd_range, 1)
        spike_rate = max(spike_rate, 0.0001)
        spike_gap = round((1 / spike_rate) * 100)
        loihi_spike_gaps[4] = spike_gap
    else:
        spike_rate = min(abs(robot_state[3]) / angular_spd_range, 1)
        spike_rate = max(spike_rate, 0.0001)
        spike_gap = round((1 / spike_rate) * 100)
        loihi_spike_gaps[5] = spike_gap
    '''
    Obstacle
    '''
    robot_scan[robot_scan == 0] = 0.0001
    spike_gap = np.round((1 / robot_scan) * 100).astype(int)
    spike_gap = spike_gap.tolist()
    loihi_spike_gaps[6:] = spike_gap
    return loihi_spike_gaps


def robot_state_2_loihi_spike_prob(robot_state,
                                   robot_scan,
                                   goal_dir_range=math.pi,
                                   linear_spd_range=0.5,
                                   angular_spd_range=2.0,
                                   precision=100):
    """
    Transform Robot State (Relative Goal Direction, Linear Spd, Angular Spd) to Loihi Spike Probability
    :param robot_state: robot state
    :param robot_scan: robot scan
    :param goal_dir_range: goal dir range
    :param linear_spd_range: linear spd range
    :param angular_spd_range: angular spd range
    :param max_gap: max time gap
    :return: loihi_spike_gaps
    """
    loihi_spike_prob = [0 for _ in range(6 + len(robot_scan))]
    '''
    Goal Relative Direction
    '''
    if robot_state[0] >= 0:
        loihi_spike_prob[0] = min(robot_state[0] / goal_dir_range, 1)
    else:
        loihi_spike_prob[1] = min(abs(robot_state[0]) / goal_dir_range, 1)
    '''
    Goal Distance
    '''
    loihi_spike_prob[2] = robot_state[1]
    '''
    Linear Spd
    '''
    loihi_spike_prob[3] = min(abs(robot_state[2]) / linear_spd_range, 1)
    '''
    Angular Spd
    '''
    if robot_state[3] >= 0:
        loihi_spike_prob[4] = min(robot_state[3] / angular_spd_range, 1)
    else:
        loihi_spike_prob[5] = min(abs(robot_state[3]) / angular_spd_range, 1)
    '''
    Obstacle
    '''
    loihi_spike_prob[6:] = robot_scan.tolist()
    for num in range(len(loihi_spike_prob)):
        loihi_spike_prob[num] = round(loihi_spike_prob[num] * precision)
    return loihi_spike_prob


def loihi_spikes_2_robot_action(out_spikes, window=50, rand_e=0.05,
                                wheel_max=0.5, wheel_min=0.05, diff=0.25):
    """
    Decode Loihi Spikes for Wheels to Linear and Angular Speed to Robot
    :param out_spikes: Loihi output spikes
    :param window: window size for one step
    :param rand_e: random e value
    :param wheel_max: wheel spd max
    :param wheel_min: wheel spd min
    :param diff: wheel diff distance
    :return: action
    """
    raw_action = np.array(out_spikes) / window
    noise = np.random.randn(2)
    raw_action = (1 - rand_e) * raw_action + rand_e * noise
    raw_action = np.clip(raw_action, [0., 0.], [1., 1.])
    l_spd = raw_action[0] * (wheel_max - wheel_min) + wheel_min
    r_spd = raw_action[1] * (wheel_max - wheel_min) + wheel_min
    linear = (l_spd + r_spd) / 2
    angular = (r_spd - l_spd) / diff
    return [linear, angular]


def combine_multiple_into_one_int(input_list, num_bits=7, overall_bits=28):
    """
    Combine multiple integers into one integer to save space
    :param input_list: list of input integers
    :param num_bits: max number of bits for item in input list
    :param overall_bits: overall bits of one integer
    :return: encode_list
    """
    int_per_int_num = overall_bits // num_bits
    assert (len(input_list) % int_per_int_num) == 0
    encode_list = []
    encode_list_num = len(input_list) // int_per_int_num
    for num in range(encode_list_num):
        start_num = num * int_per_int_num
        end_num = (num + 1) * int_per_int_num
        big_int = 0
        for i, small_int in enumerate(input_list[start_num:end_num], 0):
            big_int = big_int + (small_int << (i * num_bits))
        encode_list.append(big_int)
    return encode_list


def decoder_multiple_from_one_int(encode_list, num_bits=7, overall_bits=28):
    """
    Decode one integer into multiple integers
    :param encode_list: list of encoded integers
    :param num_bits: max number of bits for item in input list
    :param overall_bits: overall bits of one integer
    :return: input_list
    """
    input_list = []
    int_per_int_num = overall_bits // num_bits
    for big_int in encode_list:
        tmp_big_int = big_int
        for i in range(int_per_int_num):
            small_int = tmp_big_int - (tmp_big_int >> num_bits << num_bits)
            tmp_big_int = tmp_big_int >> num_bits
            input_list.append(small_int)
    return input_list


if __name__ == '__main__':
    in_list = [100, 111, 127, 120, 95, 60, 0, 37]
    en_list = combine_multiple_into_one_int(in_list)
    de_list = decoder_multiple_from_one_int(en_list)
    print(in_list, en_list, de_list)
