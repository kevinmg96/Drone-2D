"""
Sistema: 1 drone , multiples objetivos. dar servicio a dos usuarios en tierra.

algoritmo de optimizacion: Q networks. campo de accion: discreto, discretizar angulos direccion, discretizar distancia maxima de desplazamiento.

codigo de training/testing del agente en un entorno de tf usando la libraria tf-agents.
usaremos replay buffer reverb, DQN vease: Train a Deep Q Netowrk with TF- Agents
"""

#-------------- libraries -------------------------------------------------------------------------- #######################################################

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
import rps.Modules.myenv_tf_agents as myenv_tf_agents

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
import glob
import re
import os
import h5py
import json

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy,epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
import reverb
import tf_agents

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential,load_model,model_from_json
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error
import keras




#-------------- libraries -------------------------------------------------------------------------- #######################################################


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
gamma =0.995

#reward characteristics...
weight_data_rate = 3.0
weight_rel_dist = 0.09
penalize_drone_out_range = 1.0

train_max_iter = 600
#----------------------------------------------DQN agent characteristics ----------------------------------------------------------#

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()

#creamos proceso movilidad y transmision de data gus
obj_process_mob_trans_gu = gu_process.ProcesGuMobility()
obj_process_mob_trans_gu.setStopProcess()

train_py_robotarium = environment.environment(boundaries,initial_conditions,state_dimension,cartesian_action,gamma,obj_process_mob_trans_gu,show_figure=show_figure,sim_in_real_time=True,
    Rc = rc, FaceColor = rc_color,PoseGu = arr_gu_pose,GuRadius = graph_rad,GuColorList = list_color_gus,
       PlotDataRate = True, MaxGuDist = max_gu_dist, MaxGuData = max_gu_data, StepGuData = step_gu_data,PositionController = unicycle_position_controller,
                            RewardFunc = myenv_tf_agents.rewardFunc4,
                            WeightDataRate = weight_data_rate,
                            WeightRelDist = weight_rel_dist,
                            PenalDroneOutRange = penalize_drone_out_range  )

eval_py_robotarium = environment.environment(boundaries,initial_conditions,state_dimension,cartesian_action,gamma,obj_process_mob_trans_gu,show_figure=show_figure,sim_in_real_time=True,
    Rc = rc, FaceColor = rc_color,PoseGu = arr_gu_pose,GuRadius = graph_rad,GuColorList = list_color_gus,
       PlotDataRate = True, MaxGuDist = max_gu_dist, MaxGuData = max_gu_data, StepGuData = step_gu_data,PositionController = unicycle_position_controller,
                            RewardFunc = myenv_tf_agents.rewardFunc4,
                            WeightDataRate = weight_data_rate,
                            WeightRelDist = weight_rel_dist,
                            PenalDroneOutRange = penalize_drone_out_range  )


train_tf_robotarium = tf_py_environment.TFPyEnvironment(train_py_robotarium)
eval_tf_robotarium = tf_py_environment.TFPyEnvironment(eval_py_robotarium)

#------------------------------- SETTING WORKING PATH AND PROCESSORS ----------------------------- ###################################################
#working path setup based on os and pc ..
if platform == "linux":
      working_path = "/mnt/c/"
else: #windows
      working_path = "C:/"

working_directory = ["Users/CIMB-WST/Documents/Kevin Javier Medina Gómez/Tesis/1 Drone 2D GUs/robotarium_python_simulator",
"Users/kevin/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/Tesis/Project Drone 2D/Drone-2D",
"Users/opc/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/Tesis/Project Drone 2D/Drone-2D"]

trained_premodel_path = working_path + working_directory[0] + "/rps/NN_models/Trained/DQN single agent-multi objective/02_11_2023/model 1 v1/"

#model = load_model(trained_premodel_path + 'saved_model.pb',compile = False)

#setting to work with CPU or GPU...
bool_use_gpu = True

if  not bool_use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#------------------------------- SETTING WORKING PATH AND PROCESSORS ----------------------------- ###################################################


# Rest of Hyperparameters
num_iterations = 1000000 # @param {type:"integer"}

initial_collect_steps = train_max_iter  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 500000  # @param {type:"integer"}
n_step_update = 2

batch_size = 5000  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 4500  # @param {type:"integer"}
training_interval = 50 #intervalos de episodios para ejecutar el entrenamiento

num_eval_episodes = 100  # @param {type:"integer"}
eval_interval = 15000  # @param {type:"integer"}

bool_first_training_model = True #primera vez que entenaremos este modelo

# fully connected layer architecture
fc_layer_params = (128,56,12 )
output_layer_activation_function = keras.activations.linear

# defining q network using train_env and fully connected layer architecture
q_net = q_network.QNetwork(
    train_tf_robotarium.observation_spec(),
    train_tf_robotarium.action_spec(),
    fc_layer_params=fc_layer_params,
    q_layer_activation_fn=output_layer_activation_function
)

# adam optimizer to do the training
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

#train_step_counter = tf.Variable(0)
global_step = tf.compat.v1.train.get_or_create_global_step()

# create agent using the network, specifications, and other parameters
agent = dqn_agent.DqnAgent(
    train_tf_robotarium.time_step_spec(),
    train_tf_robotarium.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=global_step,epsilon_greedy= 0.1
)

# initialize the agent
agent.initialize()

# random policy -- select action from the list of actions randomly
random_policy = random_tf_policy.RandomTFPolicy(train_tf_robotarium.time_step_spec(),
                                                train_tf_robotarium.action_spec())



#----------------------------------------------------------- DATA COLLECTION --------------------------------------------------------- ############


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

debug_driver = False
buffer = []
metric = py_metrics.AverageReturnMetric()


if bool_first_training_model: #primera vez que entrenaremos este modelo
    py_driver.PyDriver(
        train_py_robotarium,
        py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
        [rb_observer] if not debug_driver else [rb_observer] + [buffer.append,metric],
        max_episodes=initial_collect_steps).run(train_py_robotarium.reset())

    if debug_driver:
        print('Replay Buffer:')
        for traj in buffer:
            print(traj)

        print('Average Return: ', metric.result())


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

#print(f"tf network weights prior loading checkpoint : {agent._q_network.get_weights()}")

#-------- creamos checkpointer object para guardar los modelos entrenados
train_checkpointer = common.Checkpointer(
    ckpt_dir=trained_premodel_path,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

      
#print(f"tf network weights after loading checkpoint : {agent._q_network.get_weights()}")

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)


#----------------------------------------------------------- DATA COLLECTION --------------------------------------------------------- ############
#---------------------------------------------------------- TRAINING AGENT -------------------------------------------------------------- #########



# Evaluate the agent's policy once before training.
#avg_return = 0.0#myenv_tf_agents.compute_avg_return(eval_tf_robotarium, agent.policy, num_eval_episodes)
returns = []

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    train_py_robotarium,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

"""
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)
"""

for i in range(1,num_iterations + 1):
    #number of episodes

    #Reset the environment at the beginning of the episode.
    time_step = train_py_robotarium.reset()

    while not time_step.is_last():

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        #step = agent.train_step_counter.numpy()

    if i % training_interval == 0:

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss  

    if i % log_interval == 0:
        print('step = {0}: loss = {1}'.format(i, train_loss))

    if i % eval_interval == 0:
        avg_return = myenv_tf_agents.compute_avg_return(eval_tf_robotarium, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(i, avg_return))
        returns.append(avg_return)

#save model using checkpointer

train_checkpointer.save(global_step)

#create keras neural network
q_network_keras  = myenv_tf_agents.createNetwork(state_dimension,fc_layer_params,cartesian_action.shape[0],output_layer_activation_function)

#save tf weights into keras model
myenv_tf_agents.save_tf_weights_to_keras_model_weights(agent._policy.variables(),q_network_keras,trained_premodel_path + "model_1_v1_weights_1.keras")



#---------------------------------------------------------- TRAINING AGENT -------------------------------------------------------------- #########


# ------------------------------------------------------------ VISUALIZATION ------------------------------------------------------------ ######

steps = range(1, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.savefig(trained_premodel_path + "model_1_v1--1.png")
plt.show()

# ------------------------------------------------------------ VISUALIZATION ------------------------------------------------------------ ######



#------------------------------------------------------ EVALUATING POLICIY ------------------------------------------------------------------- ###
#myenv_tf_agents.create_policy_eval_video(agent.policy,trained_premodel_path + "trained-agent",eval_tf_robotarium,eval_py_robotarium,5)

#------------------------------------------------------ EVALUATING POLICIY ------------------------------------------------------------------- ###
