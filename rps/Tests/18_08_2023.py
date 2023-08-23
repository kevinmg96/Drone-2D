"""
Sistema: 1 drone , multiples objetivos. dar servicio a dos usuarios en tierra.

algoritmo de optimizacion: Q networks. campo de accion: discreto, discretizar angulos direccion, discretizar distancia maxima de desplazamiento.

"""
import rps.Modules.environment as environment
import rps.Modules.gu as gu
import rps.Modules.misc as misc
import rps.Modules.Process as gu_process
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

# Instantiate Robotarium object
N = 5
initial_conditions = np.array(np.mat('0.25 0.5 0.75 1 1.25; 0.2 0.5 0.75 1.0 1.25; 0 0 0 0 0'))
#r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)
#dimensiones ambiente (punto origen x, punto origen y, ancho, alto)
boundaries = [0,0,3.2,2.0]


r = environment.environment(boundaries,number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

#GU characteristics
max_gu_dist = 0.2#m
list_color_gus = ["k","r"]
num_gus = 2
x = 0.5
y = 0.5
gu_pos = (0.125,.95)
fac = 0.25
graph_rad = 0.12 #en metros, general para todos
max_gu_data = 100 #bytes/s, kbytes/s
step_gu_data = 10 

list_gu_pose = [(gu_pos[0] + fac * i,gu_pos[1] + fac * i,0) for i in range(num_gus)]
obj_list_gus = r.createGUs(Pose = list_gu_pose, Radius = graph_rad, FaceColor = list_color_gus)


# Define goal points by removing orientation from poses
#inclui random goal points gus
goal_points_robots = np.array(np.mat('0.35 1.2 0.95 1.4 2.5; 1.0 0.25 1.75 1.5 0.15; 0 0 0 0 0'))#generate_initial_conditions(N)
goal_points_gus = np.array(np.mat('0.75 2.3; 0.35 0.47; 0 0'))#generate_initial_conditions(num_gus)

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()

# Create barrier certificates to avoid collision
#uni_barrier_cert = create_unicycle_barrier_certificate()

#creamos proceso movilidad y transmision de data gus
obj_process_mob_trans_gu = gu_process.ProcesGuMobility()

process_mob_trans_gu = threading.Thread(target= obj_process_mob_trans_gu.guProcess, args=(r,obj_list_gus,max_gu_dist,unicycle_position_controller,
                                                          at_pose,False))

#ejecutamos proceso gu
process_mob_trans_gu.start()

# define x initially for robots 
x_robots = r.get_poses()

r.step_v2(obj_list_gus,True)





while True:
    #get goal points randomly for robots and gus
    obj_process_mob_trans_gu.pauseProcess(True)

    obj_process_mob_trans_gu.pauseProcess()

    obj_process_mob_trans_gu.pauseProcess(True)

    obj_process_mob_trans_gu.stop_mobility_event.set()


    

    """
    # While the number of robots at the required poses is less
    # than N... 
    while ((np.size(at_pose(x_robots, goal_points_robots, position_error = 0.05, rotation_error=100)) +\
             np.size(at_pose(x_gus, goal_points_gus,position_error = 0.05, rotation_error=100))) != N + num_gus):

        # Get poses of agents
        x_robots = r.get_poses()
        x_gus = gu.GroundUser.get_gu_poses(obj_list_gus)

        # Create single-integrator control inputs for mobile agents and gus
        dxu_robots = unicycle_position_controller(x_robots, goal_points_robots[:2][:])
        dxu_gus = unicycle_position_controller(x_gus, goal_points_gus[:2,:])

        # Create safe control inputs (i.e., no collisions)
        #dxu_robots = uni_barrier_cert(dxu_robots, x_robots)

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu_robots,True)
        r.set_velocities(np.arange(num_gus), dxu_gus)
        # Iterate the simulation
        r.step_v2(obj_list_gus)
    """
#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
