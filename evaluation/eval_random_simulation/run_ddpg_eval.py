import rospy
import sys
sys.path.append('../../')
from evaluation.eval_random_simulation.rand_eval_gpu import RandEvalGpu
from evaluation.eval_random_simulation.utility import *


def evaluate_ddpg(pos_start=0, pos_end=199, model_name='ddpg', save_dir='../saved_model/',
                  state_num=22, is_scale=True, is_poisson=False, is_save_result=False,
                  use_cuda=True):
    """
    Evaluate DDPG in Simulated Environment

    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param state_num: shape of input state
    :param is_scale: if true normalize input state
    :param is_poisson: if true use Poisson encoding for input state
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    """
    rospy.init_node('ddpg_eval')
    poly_list, raw_poly_list = gen_test_env_poly_list_env()
    start_goal_pos = pickle.load(open("eval_positions.p", "rb"))
    robot_init_list = start_goal_pos[0][pos_start:pos_end + 1]
    goal_list = start_goal_pos[1][pos_start:pos_end + 1]
    net_dir = save_dir + model_name + '.pt'
    actor_net = load_test_actor_network(net_dir, state_num=state_num)
    eval = RandEvalGpu(actor_net, robot_init_list, goal_list, poly_list,
                       max_steps=1000, action_rand=0.01, goal_dis_min_dis=0.3,
                       is_scale=is_scale, is_poisson=is_poisson, use_cuda=use_cuda)
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
    parser.add_argument('--poisson', type=int, default=0)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    IS_POISSON = False
    STATE_NUM = 22
    MODEL_NAME = 'ddpg'
    if args.poisson == 1:
        IS_POISSON = True
        STATE_NUM = 24
        MODEL_NAME = 'ddpg_poisson'
    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    evaluate_ddpg(use_cuda=USE_CUDA, state_num=STATE_NUM,
                  is_poisson=IS_POISSON, model_name=MODEL_NAME,
                  is_save_result=SAVE_RESULT)
