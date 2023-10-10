import numpy as np
import rps.robotarium as robotarium

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rps.Modules.gu as gu
import rps.Modules.mobileagent as mobileagent
import rps.Modules.misc as misc


class environment(robotarium.Robotarium):
    """
    esta superclase extendiende la funcionalidad del robotarium para poder trabajar con un ambiente integrado con GUs,
    así como sus respectivas clases
    """
    def __init__(self,boundaries, initial_conditions, show_figure=True, sim_in_real_time=True,**kwargs):
        super().__init__(boundaries,initial_conditions,show_figure, sim_in_real_time,Rc = kwargs["Rc"],
        FaceColor = kwargs["FaceColor"], PoseGu = kwargs["PoseGu"], GuRadius = kwargs["GuRadius"], GuColorList = kwargs["GuColorList"],
        PlotDataRate = kwargs["PlotDataRate"], MaxGuDist = kwargs["MaxGuDist"],MaxGuData = kwargs["MaxGuData"],
        StepGuData = kwargs["StepGuData"])


    def stepEnv(self,action,pose_eval,unicycle_position_controller,reward_func,weight_dr,weight_dis,drone_boundaries_penalization,
                bool_debug = False):
        """
        esta funcion ejecutara la accion previamente seleccionada, es decir movera al drone a la accion deseada.
        retornaremos reward, nuevo estado, si es terminal.
        """
        #formato de accion: (magnitud,direccion) ->convertir en punto
        possible_drone_next_pos,_ = misc.computeNextPosition(action[0],self.obj_drones.poses[:2,0],action[1],False)
        possible_drone_next_pos = possible_drone_next_pos.reshape(-1,1)

        terminal_state = self.isTerminalState(possible_drone_next_pos)
        if bool_debug:
            print("Drone next possible action: magnitude : {}, direction : {}".format(action[0],action[1]))
            print("Drones current possible position: {}".format(self.obj_drones.poses[:2,0]))
            print("is terminal state ?: {}".format(terminal_state))

        if terminal_state:
            #keep drone current position
            flag_drone_out_range = True
            drone_next_pos = self.obj_drones.poses[:2,0]
        else:
            #move drone to next position
            flag_drone_out_range = False

            drone_next_pos = possible_drone_next_pos[:,0]

        if bool_debug:
            print("final drones next position: {}".format(drone_next_pos))
        if self.show_figure: #if debugging results, use single integrator to drive the dynamics of the robots
            # Get poses of agents
            x_robots = self.obj_drones.poses

            goal_points_robots = np.zeros([3,self.number_of_robots])
            goal_points_robots[:2,0] = drone_next_pos
            goal_points_robots[2,0] = x_robots[2,0]

            while (np.size(pose_eval(x_robots, goal_points_robots, position_error = 0.05, rotation_error=300)) != self.number_of_robots ):

            
                # Create single-integrator control inputs for mobile agents and gus
                dxu_robots = unicycle_position_controller(x_robots, goal_points_robots[:2][:])

                # Set the velocities by mapping the single-integrator inputs to unciycle inputs
                self.set_velocities(dxu_robots,True)

                # Iterate the simulation
                self.step_v2(True)

                # Get poses of agents
                x_robots = self.obj_drones.poses

        else: # if training, the next drone positions will be automatically used to update drones pose
            self.obj_drones.poses[:2,0] = drone_next_pos

        #actualizamos los registros del drone...
        self.obj_drones.echoRadio(self.obj_gus)

        #return the transition tuple -> next_state,reward,is_next_state_terminal
        reward = self.getReward(reward_func,weight_dr,weight_dis,flag_drone_out_range,drone_boundaries_penalization)
        next_state = self.getState()

        drone_gu_pose = np.concatenate([self.obj_drones.poses,self.obj_gus.poses],axis = 1)
        is_next_state_terminal = self.isTerminalState(drone_gu_pose)
        return next_state,reward,is_next_state_terminal

    def resetEnv(self,factor_gu_out_rc = 1.35):
        """
        resetereamos la posicion del robot inicial a una random.
        asi como nuevas posiciones random de los GUs
        fecha actualizacion 04/09/2023. actualice la metodologia para resetear las posiciones de los agentes del ambiente.
        Pos drone en ambiente seguira siendo random. sin embargo, las posiciones del gu seran distribuidas de dos maneras diferentes,
        una distribucion de probabilidad decidira si colocar a los gus de manera aleatoria por el terreno, o dentro
        del rango rc del drone
        """
        #reseteamos la figura y creamos visualmente de nuevo al robot en una posicion random.
        self.reset_env = True
        #random position robot...

        random_new_pos_robots = np.random.random(size=(3,self.obj_drones.poses.shape[1]))
        random_new_pos_robots[0,:] = random_new_pos_robots[0,:] * self.boundaries[2]
        random_new_pos_robots[1,:] = random_new_pos_robots[1,:] * self.boundaries[3]
        random_new_pos_robots[2,:] = 0.0

        #update robot poses in environment objects...
        self.obj_drones.poses = random_new_pos_robots

        #reset drones velocities
        self.obj_drones.reset_drones_velocities()

        #create drones in new position..     y reseteamos listas con data de drones 
        self.generateVisualRobots()

        #initialize position robots...
        if self.show_figure:     # training, execute step   
            self.step_v2(True)

        #--------------------------------------------------

        #randomly select a distribution, based
        next_reset_distribution = [misc.randomChoice(1.0),misc.poissonChoice(1.0,5),
        misc.gaussianChoice(1.0)]
        p_distribution = [0.3,0.4,0.3]
        next_reset_env = np.random.choice(next_reset_distribution,p=p_distribution)

        bool_random_gus = np.where(next_reset_env < 0.5,True,False)
   

        if bool_random_gus:
            #place gus randomly in environment
            for i in range(self.obj_gus.poses.shape[1]):
                self.obj_gus.poses[0,i] = np.random.random() * self.boundaries[2]
                self.obj_gus.poses[1,i] = np.random.random() *  self.boundaries[3]
        else:
            #random position gus inside drone rc...
            for i in range(self.obj_gus.poses.shape[1]):
                next_pos_gus_inside_rc = np.where(misc.poissonChoice(1.0,5) < 0.5,True,False)
                if next_pos_gus_inside_rc: #gu next position inside rc...
                    next_pos_gus_mag = misc.gaussianChoice(self.obj_drones.rc,0.25)                
                else: #gu next position outside rc...
                    next_pos_gus_mag = self.obj_drones.rc * factor_gu_out_rc
                self.obj_gus.poses[:2,i],_ = misc.computeNextPosition(next_pos_gus_mag,self.obj_drones.poses[:2,0])
            
                
        #reset gus  velocities
        self.obj_gus.reset_gus_velocities()  
 
        self.resetGUs()
        
        #initialize position GU...
        if self.show_figure: # training, execute step         
            self.step_v2()

        self.reset_env = False


    def isTerminalState(self,array_poses,tolerance = 0.01):
        """
        esta funcion evaluara si el estado actual del ambiente es terminal.
        por el momento el estado terminal sera si el drone navega fuera de los limites del espacio 2d o
        algun gu navega fuera del ambiente
        """
        #drone_gu_pose = np.concatenate([self.obj_drones.poses,self.obj_gus.poses],axis = 1)

        c_1 = (array_poses[0,:] > self.boundaries[2]).any()
        c_2 = (array_poses[0,:] < self.boundaries[0]).any()
        c_3 = (array_poses[1,:] > self.boundaries[3]).any()
        c_4 = (array_poses[1,:] < self.boundaries[1]).any()
        return np.where(c_1 or c_2 or c_3 or c_4,True,False)

    def getReward(self,reward_func,weight_dr,weight_dis,flag_drone_out_range,drone_boundaries_penalization,bool_debug = False):
        """
        esta funcion implementara la ecuacion para obtener la recompensa por la accion ejecutada del drone.
        R = conec_1 * data_rate_1 + conec_2 * data_rate_2
        """
        reward = reward_func(self, weight_dr,weight_dis)
        if flag_drone_out_range:
            reward -= drone_boundaries_penalization

        if bool_debug:
            print("reward : {}".format(reward))
        return reward
    
        #reward = (self.obj_gus.transmission_rate[0]/trans_rate_tot) * ((self.obj_drones.rc - dis_1) / self.obj_drones.rc) +\
        #(self.obj_gus.transmission_rate[1]/trans_rate_tot) * ((self.obj_drones.rc - dis_2) / self.obj_drones.rc)
        #return conec_1 * (self.obj_gus.transmission_rate[0]/trans_rate_tot) + conec_2 * (self.obj_gus.transmission_rate[1]/trans_rate_tot)

    def getState(self):
        """
        esta funcion nos permitira obtener el estado global del sistema. 
        (Pose drone, Pose GUs, isGUcovered, GU tranmission rate

        actualizacion fecha: 30/08/2023: obtenedremos un vector con las siguientes salidas, para la red neuronal:
        1.- distancia entre drone y gu 1
        2.- bool conectividad drone y gu 1
        3.- data rate drone y gu 1
        4.- distancia entre drone y gu 2
        5.- bool conectividad drone y gu 2
        6.- data rate drone y gu 2

        normalizaremos los valores entre [0 y 1] para los valores de data rate
        """  
        env_state = np.zeros([1,6])
        env_state[0,0] = self.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        env_state[0,1] = self.obj_drones.dict_gu[0]["Gu_0"]["Connection"]
        env_state[0,2] = np.interp(self.obj_gus.transmission_rate[0], [0,self.obj_gus.max_gu_data],[0, 1])
        env_state[0,3] = self.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        env_state[0,4] = self.obj_drones.dict_gu[0]["Gu_1"]["Connection"]
        env_state[0,5] = np.interp(self.obj_gus.transmission_rate[1], [0,self.obj_gus.max_gu_data],[0, 1])

       

        #tuple_gu_pos = tuple((gu.pose for gu in obj_gu_list))
        #tuple_gu_dic = (obj_drone_list[0].dict_gu)
        
        return env_state#(obj_drone_list[0].pose, tuple_gu_pos, tuple_gu_dic)
        

    def getAreaDimensions(self):
        #retornamos punto origen, width, height
        return self.boundaries
    
    




    

        
    



