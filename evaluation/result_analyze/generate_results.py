import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from evaluation.result_analyze.utility import analyze_run, plot_robot_paths
from evaluation.eval_random_simulation.utility import gen_test_env_poly_list_env

MODEL_NAME = 'sddpg_bw_5'
# MODEL_NAME = 'ddpg'
# MODEL_NAME = 'ddpg_poisson'
FILE_NAME = MODEL_NAME + '_0_199.p'

run_data = pickle.load(open('../record_data/' + FILE_NAME, 'rb'))
s_list, p_dis, p_time, p_spd = analyze_run(run_data)
print(MODEL_NAME + " random simulation results:")
print("Success: ", s_list[0], " Collision: ", s_list[1], " Overtime: ", s_list[2])
print("Average Path Distance of Success Routes: ", np.mean(p_dis[p_dis > 0]), ' m')
print("Average Path Time of Success Routes: ", np.mean(p_time[p_dis > 0]), ' s')
# print("Average Path Speed of Success Routes: ", np.mean(p_spd[p_dis > 0]), ' m/s')


start_goal_pos = pickle.load(open("../eval_random_simulation/eval_positions.p", "rb"))
goal_list = start_goal_pos[1]
poly_list, raw_poly_list = gen_test_env_poly_list_env()
plot_robot_paths(run_data, raw_poly_list, goal_list)
plt.show()
