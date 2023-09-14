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

import numpy as np
import time
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import time
import threading
import itertools
import pickle

# -----------------------------------------TEST MODEL ----------------------------------------------------------------------#

def test_performance_model(dqn_agent,env,**args):
    #reseteamos el ambiente, al inicio de cada episodio...
    env.resetEnv()
           
    #actualizamos los registros del drone...
    env.obj_drones.echoRadio(env.obj_gus)

    #get initial state
    current_state = env.getState()
    is_next_state_terminal = False

    rewards_per_episode = []
    counter_train_timeslot = 0

    while not is_next_state_terminal: #mientras no hemos llegado a un estado terminal, continuar iterando avanzando en el episodio
        counter_train_timeslot += 1
        if counter_train_timeslot >dqn_agent.timeslot_train_iter_max: #salimos del inner loop. cambiamos a otro ep
            break                

        #seleccionamos accion para agente de acuerdo a e greedy strategy
        index_action, action = dqn_agent.selectAction(current_state)

        #ejecutamos accion del DQN agent (desplazamos al drone...), retornamos new_state,reward,is_terminal_state
        next_state,reward,is_next_state_terminal = env.stepEnv(action,at_pose,args["PositionController"])
        rewards_per_episode.append(reward)     

        current_state = next_state
                
        # ejecutamos acciones de los gus...
        obj_process_mob_trans_gu.guProcess(env,args["PositionController"],at_pose,False)

        #actualizamos los registros del drone...
        env.obj_drones.echoRadio(env.obj_gus)   

        #si el nuevo estado del ambiente es terminal, un gu se desplazo fuera de los limites, terminamos este episodio
        if env.isTerminalState():
            break 

    #plot rewards..
    DQN.plot_rewards(rewards_per_episode)

# -----------------------------------------TEST MODEL ----------------------------------------------------------------------#

# Instantiate Robotarium object

initial_conditions = np.array(np.mat('0.75;1.0;0.0'))#np.mat('0.25 0.5 0.75 1 1.25; 0.2 0.5 0.75 1.0 1.25; 0 0 0 0 0'))

#dimensiones ambiente (punto origen x, punto origen y, ancho, alto)
boundaries = [0,0,3.2,2.0]

#--------------------------------------------Drone Characteristics ---------------------------------------------------------- #
rc = 0.4 #radio de comunicaciones en m
rc_color = "k"
drone_disp_range = [0,0.3] #rango de movimiento permitido del drone
drone_disp_num  = 5#numero de divisiones en la accion displacement

arr_drone_disp_values = np.linspace(drone_disp_range[0],drone_disp_range[1],num = drone_disp_num)
drone_angle_range = [0, 2*np.pi] #rango de direcciones
drone_angle_num = 8#numero de divisiones en la accion direction 

arr_drone_angle_values = np.linspace(drone_angle_range[0],drone_angle_range[1],num = drone_angle_num, endpoint= False)

#cartesian product (displacement, direction):
cartesian_action = np.array(list(itertools.product(arr_drone_disp_values,arr_drone_angle_values)))
encode_action = np.arange(cartesian_action.shape[0])

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

r = environment.environment(boundaries,initial_conditions=initial_conditions,show_figure=True,sim_in_real_time=True,
    Rc = rc, FaceColor = rc_color,PoseGu = arr_gu_pose,GuRadius = graph_rad,GuColorList = list_color_gus,
       PlotDataRate = True, MaxGuDist = max_gu_dist, MaxGuData = max_gu_data, StepGuData = step_gu_data  )



#----------------------------------------------DQN agent characteristics ----------------------------------------------------------#
state_dimension = 6
gamma = 0.1

#----------------------------------------------DQN agent characteristics ----------------------------------------------------------#

# Define goal points by removing orientation from poses
#inclui random goal points gus
goal_points_robots = np.array(np.mat('0.35 1.2 0.95 1.4 2.5; 1.0 0.25 1.75 1.5 0.15; 0 0 0 0 0'))
goal_points_gus = np.array(np.mat('0.75 2.3; 0.35 0.47; 0 0'))#generate_initial_conditions(num_gus)

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()

# Create barrier certificates to avoid collision
#uni_barrier_cert = create_unicycle_barrier_certificate()

#initialize position of robots...

r.step_v2(True)

#creamos proceso movilidad y transmision de data gus
obj_process_mob_trans_gu = gu_process.ProcesGuMobility()

#process_mob_trans_gu = threading.Thread(target= obj_process_mob_trans_gu.guProcess, args=(r,obj_gus_list,obj_drone_list,
#max_gu_dist, max_gu_data,step_gu_data, unicycle_position_controller,at_pose,False))

#ejecutamos proceso gu
#process_mob_trans_gu.start()

#primera ejecucion del proceso cambiar la posicion de drones...

obj_process_mob_trans_gu.setStopProcess()

pretrained_model_path = "C:/Users/kevin/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/Tesis/Project Drone 2D/Drone-2D/rps/NN_models/Pretrained/DQN single agent-objective/05_09_2023/model 1 v2/"
pretrained_model_filename = "model_1_v2--30.keras"
#load model test...
num_episodes = 2500
batch_size = 700
train_max_iter = 200
save_interval_premodel = 2000
dqn_agent = DQN.DQNAgent(state_dimension,cartesian_action,4000,gamma,prob_epsilon,num_episodes,batch_size,train_max_iter,
                         save_interval_premodel,pretrained_model_path + pretrained_model_filename)



# test model...
test_performance_model(dqn_agent,r, PositionController = unicycle_position_controller)
