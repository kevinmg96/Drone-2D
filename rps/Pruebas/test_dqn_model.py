"""
Probaremos la red neuronal Q network entrenada usando DQN
"""
import rps.Modules.DQN as DQN

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

import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential,load_model


# -----------------------------------------TEST MODEL ----------------------------------------------------------------------#

def test_performance_model(dqn_agent,env):
    while True:
        #reseteamos el ambiente, al inicio de cada episodio...
        env.resetEnv()

        #set transmission rate for each gu...
        #actualizamos estatus data tranmission y rate de los gus
        env.obj_gus.setTransmissionRate(env.obj_gus.max_gu_data,False)
        for k in range(env.obj_gus.poses.shape[1]):              
                        
            #actualizamos data transmission value en ambiente
            if env.show_figure:
                env.updateGUDataRate(k,env.obj_gus.transmission_rate[k])
        
        #actualizamos los registros del drone...
        env.obj_drones.echoRadio(env.obj_gus)

        #get initial state
        current_state = env.getState()
        is_next_state_terminal = False

        rewards_per_episode = []

        while not is_next_state_terminal: #mientras no hemos llegado a un estado terminal, continuar iterando avanzando en el episodio
            
            #seleccionamos accion para agente de acuerdo a e greedy strategy
            index_action, action = dqn_agent.selectAction(current_state)

            #ejecutamos accion del DQN agent (desplazamos al drone...), retornamos new_state,reward,is_terminal_state
            next_state,reward,_ = env.stepEnv(action,at_pose,env.kwargs_step_env["PositionController"],
                env.kwargs_step_env["RewardFunc"],env.kwargs_step_env["WeightDataRate"],env.kwargs_step_env["WeightRelDist"],
                env.kwargs_step_env["PenalDroneOutRange"])
            
            # ejecutamos acciones de los gus
            env.obj_process_mob_trans_gu.guProcess(env,env.kwargs_step_env["PositionController"],at_pose,False)

            #actualizamos los registros del drone...
            env.obj_drones.echoRadio(env.obj_gus)   

            #si el nuevo estado del ambiente es terminal, un gu se desplazo fuera de los limites, terminamos este episodio
            drone_gu_pose = np.concatenate([env.obj_drones.poses,env.obj_gus.poses],axis = 1)
            if env.isTerminalState(drone_gu_pose): #gu se desplazo fuera del area
                is_next_state_terminal = True

            rewards_per_episode.append(reward)     

            current_state = next_state
                    


        #plot rewards..
        #DQN.plot_rewards(rewards_per_episode)

        is_next_state_terminal = False

# -----------------------------------------TEST MODEL ----------------------------------------------------------------------#

# Instantiate Robotarium object

initial_conditions = np.array(np.mat('0.75;1.0;0.0'))#np.mat('0.25 0.5 0.75 1 1.25; 0.2 0.5 0.75 1.0 1.25; 0 0 0 0 0'))

#dimensiones ambiente (punto origen x, punto origen y, ancho, alto)
boundaries = [0,0,3.2,2.0]
show_figure = True
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
prob_epsilon = 0.0
prob_epsilon_decay = 0.999 #despues de ciertas iteraciones, reducimos el valor de la prob de exploracion
number_epsilon_iter_decay = 200 #cada 200 episodios reducimos el valor de epsilon

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
gamma =.995

#reward characteristics...
weight_data_rate = 5
weight_rel_dist = 0.15
penalize_drone_out_range = 1

train_max_iter = 30
#----------------------------------------------DQN agent characteristics ----------------------------------------------------------#

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()

#creamos proceso movilidad y transmision de data gus
obj_process_mob_trans_gu = gu_process.ProcesGuMobility()
obj_process_mob_trans_gu.setStopProcess()

r = environment.environment(boundaries,initial_conditions,state_dimension,cartesian_action,gamma,obj_process_mob_trans_gu,show_figure=show_figure,sim_in_real_time=True,
    Rc = rc, FaceColor = rc_color,PoseGu = arr_gu_pose,GuRadius = graph_rad,GuColorList = list_color_gus,
       PlotDataRate = True, MaxGuDist = max_gu_dist, MaxGuData = max_gu_data, StepGuData = step_gu_data,PositionController = unicycle_position_controller,
                            RewardFunc = myenv_tf_agents.rewardFunc3,
                            WeightDataRate = weight_data_rate,
                            WeightRelDist = weight_rel_dist,
                            PenalDroneOutRange = penalize_drone_out_range  )


#process_mob_trans_gu = threading.Thread(target= obj_process_mob_trans_gu.guProcess, args=(r,obj_gus_list,obj_drone_list,
#max_gu_dist, max_gu_data,step_gu_data, unicycle_position_controller,at_pose,False))

#ejecutamos proceso gu
#process_mob_trans_gu.start()


if platform == "linux":
      working_path = "/mnt/c/"
else: #windows
      working_path = "C:/"

working_directory = ["Users/CIMB-WST/Documents/Kevin Javier Medina Gómez/Tesis/1 Drone 2D GUs/robotarium_python_simulator",
"Users/kevin/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/Tesis/Project Drone 2D/Drone-2D",
"Users/opc/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/Tesis/Project Drone 2D/Drone-2D"]

pretrained_model_path = working_path + working_directory[0] + "/rps/NN_models/Trained/DQN single agent-multi objective/29_10_2023/model 1 v6/"
pretrained_model_filename = "model_1_v6_weights.keras"


#data = DQN.load_info_data(pretrained_model_path + "model_1_v2_data.txt")
#DQN.plot_rewards(data)
#load model test...

num_episodes = 2500
batch_size = 500

save_interval_premodel = 2000

hid_layer_neurons = (128,56,12 )
output_layer_activation_function = keras.activations.linear
dqn_agent = DQN.DQNAgent(state_dimension,cartesian_action,4000,gamma,prob_epsilon,num_episodes,batch_size,hid_layer_neurons,output_layer_activation_function,
                         train_max_iter,save_interval_premodel,None,pretrained_model_path + pretrained_model_filename)



# test model...
test_performance_model(dqn_agent,r)

