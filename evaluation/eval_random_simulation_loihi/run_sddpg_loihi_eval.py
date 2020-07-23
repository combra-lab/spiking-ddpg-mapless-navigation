import rospy
import sys
import pickle
sys.path.append('../../')
from evaluation.loihi_network.spiking_actor_net_loihi import SpikingActorNet
from evaluation.eval_random_simulation_loihi.rand_eval_loihi import RandEvalLoihi
from evaluation.eval_random_simulation.utility import *


def evaluate_loihi_sddpg(pos_start=0, pos_end=49, model_name='sddpg_bw_5', save_dir='../saved_model/',
                         batch_window=5, is_save_result=False):
    """
    Evaluate Spiking DDPG in Simulated Environment on Intel's Loihi Neuromorphic Processor

    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param batch_window: inference timesteps
    :param is_save_result: if true save the evaluation result
    """
    rospy.init_node('ddpg_snn_test')
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    start_goal_pos = pickle.load(open("eval_positions.p", "rb"))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    w_dir = save_dir + model_name + '_weights.p'
    b_dir = save_dir + model_name + '_bias.p'
    raw_weights = pickle.load(open(w_dir, 'rb'))
    raw_bias = pickle.load(open(b_dir, 'rb'))
    core_list_input = [0 for _ in range(24)]
    core_list_hidden1 = [num // 128 + 3 for num in range(256)]
    core_list_hidden2 = [num // 128 + 5 for num in range(256)]
    core_list_hidden3 = [num // 128 + 7 for num in range(256)]
    core_list_out = [2, 2]
    core_list_bias = [1, 1, 1, 1]
    core_list = [core_list_input, core_list_hidden1, core_list_hidden2, core_list_hidden3, core_list_out,
                 core_list_bias]
    snn = SpikingActorNet(raw_weights, raw_bias, core_list, bias_amp=(1, 1, 1, 1))
    board, in_channel, out_channel = snn.setup_snn(print_axon=True)
    eval = RandEvalLoihi(in_channel, out_channel, robot_init_list, goal_list, poly_list, window=batch_window,
                         max_steps=1000, goal_dis_min_dis=0.3, scan_min_dis=0.35, action_rand=0.01)
    # Start Loihi Running
    board.startDriver()
    board.run(2 ** 20, aSync=True)
    data = eval.run_ros()
    if is_save_result:
        pickle.dump(data, open('../record_data/' + model_name + '_loihi_' + str(pos_start) + '_' + str(pos_end) + '.p',
                               'wb+'))
    print(str(model_name) + " Eval on Loihi Finished (Press Ctrl-C to quit) ...")
    board.finishRun()
    board.disconnect()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--step', type=int, default=5)
    args = parser.parse_args()

    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    MODEL_NAME = 'sddpg_bw_' + str(args.step)
    evaluate_loihi_sddpg(model_name=MODEL_NAME, is_save_result=SAVE_RESULT)
