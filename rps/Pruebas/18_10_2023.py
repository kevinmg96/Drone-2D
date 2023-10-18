"""
Sistema: 1 drone , multiples objetivos. dar servicio a dos usuarios en tierra.

algoritmo de optimizacion: Q networks. campo de accion: discreto, discretizar angulos direccion, discretizar distancia maxima de desplazamiento.

codigo empleado para entrenar la red q_state action network
"""

import rps.Modules.environment as environment
import rps.Modules.gu as gu
import rps.Modules.misc as misc
import rps.Modules.Process as gu_process
import rps.Modules.DQN as DQN
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import time
import threading
import itertools
import pickle
from sys import platform

from tf_agents.environments import utils
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
import tensorflow as tf

def rewardFunc(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_dr = conec_1 * (env.obj_gus.transmission_rate[0]/trans_rate_tot) + conec_2 *(env.obj_gus.transmission_rate[1]/
                            trans_rate_tot)

        sum_dis = ((env.obj_drones.rc - dis_1) / env.obj_drones.rc) + ((env.obj_drones.rc - dis_2) / env.obj_drones.rc)

        reward = weight_dr * sum_dr + weight_dis * sum_dis
        return reward



def rewardFunc2(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_conec = conec_1 + conec_2


        sum_dr = (env.obj_gus.transmission_rate[0]/trans_rate_tot) + (env.obj_gus.transmission_rate[1]/trans_rate_tot)

        sum_dis = ((env.obj_drones.rc - dis_1) / env.obj_drones.rc) + ((env.obj_drones.rc - dis_2) / env.obj_drones.rc)
        reward = sum_conec + weight_dr * sum_dr + weight_dis * sum_dis

        return reward


def rewardFunc3(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_dr = conec_1 * (env.obj_gus.transmission_rate[0]/trans_rate_tot) + conec_2 *(env.obj_gus.transmission_rate[1]/
                            trans_rate_tot)

        sum_dis = ((env.obj_drones.rc - dis_1) / env.obj_drones.rc) + ((env.obj_drones.rc - dis_2) / env.obj_drones.rc)

        reward = weight_dr * sum_dr #+ weight_dis * sum_dis
        return reward

#-------------------------------------------------------------------------------------------





# Instantiate Robotarium object

initial_conditions = np.array(np.mat('0.75;1.0;0.0'))#np.mat('0.25 0.5 0.75 1 1.25; 0.2 0.5 0.75 1.0 1.25; 0 0 0 0 0'))

#dimensiones ambiente (punto origen x, punto origen y, ancho, alto)
boundaries = [0,0,3.2,2.0]
show_figure = False

#--------------------------------------------Drone Characteristics ---------------------------------------------------------- #
rc = 0.5 #radio de comunicaciones en m
rc_color = "k"
disp_max = 0.35
drone_disp_num  = 5#numero de divisiones en la accion displacement
drone_disp_range = [disp_max/drone_disp_num,disp_max] #rango de movimiento permitido del drone

arr_drone_disp_values = np.linspace(drone_disp_range[0],drone_disp_range[1],num = drone_disp_num)
drone_angle_range = [0, 2*np.pi] #rango de direcciones
drone_angle_num = 8#numero de divisiones en la accion direction 

arr_drone_angle_values = np.linspace(drone_angle_range[0],drone_angle_range[1],num = drone_angle_num, endpoint= False)

#cartesian product (displacement, direction):
cartesian_action = np.array(list(itertools.product(arr_drone_disp_values,arr_drone_angle_values)))

#append action hovering (mag : 0, angle : 0)
cartesian_action = np.concatenate([cartesian_action,np.zeros([1,2])],axis = 0)


#E-greedy policy
prob_epsilon = 0.2
target_network_update_interval = 300 #cada 300 timesteps actualizaremos los pesos del target network a que sean iguales
#a los del q network

#--------------------------------------------Drone Characteristics ---------------------------------------------------------- #

#----------------------------------------------GU characteristics ---------------------------------------------------------------#
max_gu_dist = 0.18#m
list_color_gus = ["r","b"]
num_gus = 2

gu_pos = (0.525,.95)
fac = 0.25
graph_rad = 0.12 #en metros, general para todos
max_gu_data = 100.0 #bytes/s, kbytes/s
step_gu_data = 5.0 

arr_gu_pose = np.random.random(size=(3,num_gus))
arr_gu_pose[0,:] = arr_gu_pose[0,:] * boundaries[2]
arr_gu_pose[1,:] = arr_gu_pose[1,:] * boundaries[3]
arr_gu_pose[2,:] = 0.0

#----------------------------------------------GU characteristics ---------------------------------------------------------------#



#----------------------------------------------DQN agent characteristics ----------------------------------------------------------#
state_dimension = 6
gamma = 0.995

#reward characteristics...
weight_data_rate = 5
weight_rel_dist = 0.025
penalize_drone_out_range = 1.5

#----------------------------------------------DQN agent characteristics ----------------------------------------------------------#

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()


num_episodes = 10
batch_size = 650
train_max_iter = 15
save_interval_premodel = 250 #number of past epochs required for saving a pretrained model
memory_capacity = 10000
debug_interval = 250 #debugging rewards and running time per episode

r = environment.environment(boundaries,initial_conditions,state_dimension,cartesian_action,gamma,train_max_iter,show_figure=show_figure,sim_in_real_time=True,
    Rc = rc, FaceColor = rc_color,PoseGu = arr_gu_pose,GuRadius = graph_rad,GuColorList = list_color_gus,
       PlotDataRate = True, MaxGuDist = max_gu_dist, MaxGuData = max_gu_data, StepGuData = step_gu_data,PositionController = unicycle_position_controller,
                            RewardFunc = rewardFunc,
                            WeightDataRate = weight_data_rate,
                            WeightRelDist = weight_rel_dist,
                            PenalDroneOutRange = penalize_drone_out_range  )

print("TimeStep Specs:", r.time_step_spec())
print("Action Specs:", r.action_spec())

tf_r = tf_py_environment.TFPyEnvironment(r)
"""
print(isinstance(tf_r, tf_environment.TFEnvironment))
print("TimeStep Specs tf:", tf_r.time_step_spec())
print("Action Specs tf:", tf_r.action_spec())
n = 5
print(f"test reset environment for {n} episodes...")

cumulative_reward = 0
for i in range(n):
    print(f"training episode : {i}")
    print("reset env...")
    time_step = r.reset()
    j = 0
    time.sleep(5)
    while not time_step.is_last():
        random_index_action = np.random.randint(cartesian_action.shape[0])
        print(f"timeslot : {j}, action  : {random_index_action}")
        get_new_action = np.array(random_index_action, dtype=np.int32)
        time_step = r.step(get_new_action)
        print(f"timeslot : {j} , timestep : {time_step}")
        cumulative_reward += time_step.reward
        j += 1
    time.sleep(10)
print(f"Final reward : {cumulative_reward}")
"""
time_step = tf_r.reset()
rewards = []
steps = []

for _ in range(num_episodes):
  episode_reward = 0
  episode_steps = 0
  while not time_step.is_last():
    random_index_action = np.random.randint(cartesian_action.shape[0])
    action = tf.constant(random_index_action, dtype= tf.int32)
    time_step = tf_r.step(action)
    episode_steps += 1
    episode_reward += time_step.reward.numpy()
  rewards.append(episode_reward)
  steps.append(episode_steps)
  time_step = tf_r.reset()

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)

           
    

        


