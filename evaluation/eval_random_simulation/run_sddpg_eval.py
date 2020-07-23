import rospy
import sys
sys.path.append('../../')
from evaluation.eval_random_simulation.rand_eval_gpu import RandEvalGpu
from evaluation.eval_random_simulation.utility import *


def evaluate_sddpg(pos_start=0, pos_end=199, model_name='sddpg_bw_5', save_dir='../saved_model/',
                   batch_window=5, is_save_result=False, use_cuda=True):
    """
    Evaluate Spiking DDPG in Simulated Environment

    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param batch_window: inference timesteps
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    """
    rospy.init_node('sddpg_eval')
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    start_goal_pos = pickle.load(open("eval_positions.p", "rb"))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    w_dir = save_dir + model_name + '_weights.p'
    b_dir = save_dir + model_name + '_bias.p'
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    actor_net = load_test_actor_snn_network(w_dir, b_dir, device, batch_window=batch_window)
    eval = RandEvalGpu(actor_net, robot_init_list, goal_list, poly_list,
                       max_steps=1000, action_rand=0.01, goal_dis_min_dis=0.3,
                       is_spike=True, use_cuda=use_cuda)
    data = eval.run_ros()
    if is_save_result:
        pickle.dump(data,
                    open('../record_data/' + model_name + '_' + str(pos_start) + '_' + str(pos_end) + '.p', 'wb+'))
    print(str(model_name) + " Eval on GPU Finished ...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--step', type=int, default=5)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    MODEL_NAME = 'sddpg_bw_' + str(args.step)
    evaluate_sddpg(use_cuda=USE_CUDA, model_name=MODEL_NAME,
                   is_save_result=SAVE_RESULT)
