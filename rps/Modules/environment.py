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
    def __init__(self,boundaries, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):
        super().__init__(boundaries,number_of_robots, show_figure, sim_in_real_time, initial_conditions)

    def stepEnv(self,dq_agent,obj_drone_list,obj_gus_list,action,pose_eval,unicycle_position_controller,rc,max_gu_data):
        """
        esta funcion ejecutara la accion previamente seleccionada, es decir movera al drone a la accion deseada.
        retornaremos reward, nuevo estado, si es terminal.
        """
        #formato de accion: (magnitud,direccion) ->convertir en punto
        drone_next_pos = misc.computeNextPosition(action[0],obj_drone_list[0].pose[:2],action[1],False)


        # Get poses of agents
        x_robots = mobileagent.MobileAgent.get_drone_poses(obj_drone_list)

        goal_points_robots = np.zeros([3,len(obj_drone_list)])
        goal_points_robots[:2,0] = drone_next_pos

        while (np.size(pose_eval(x_robots, goal_points_robots, position_error = 0.05, rotation_error=100)) != len(obj_drone_list) ):

            # Get poses of agents
            x_robots = mobileagent.MobileAgent.get_drone_poses(obj_drone_list)

            # Create single-integrator control inputs for mobile agents and gus
            dxu_robots = unicycle_position_controller(x_robots, goal_points_robots[:2][:])

            # Set the velocities by mapping the single-integrator inputs to unciycle inputs
            self.set_velocities(np.arange(len(obj_drone_list)), dxu_robots,True)

            # Iterate the simulation
            self.step_v2(obj_gus_list,obj_drone_list,True)

        #actualizamos los registros del drone...
        obj_drone_list[0].echoRadio(obj_gus_list,rc)

        #return the transition tuple -> next_state,reward,is_next_state_terminal
        reward = dq_agent.getReward(obj_drone_list,obj_gus_list)
        next_state = self.getState(obj_drone_list,obj_gus_list,max_gu_data)
        is_next_state_terminal = self.isTerminalState(obj_drone_list,obj_gus_list)
        return next_state,reward,is_next_state_terminal

    def resetEnv(self,obj_drone_list,obj_gus_list,graph_rad,list_color_gus,rc,rc_color):
        """
        resetereamos la posicion del robot inicial a una random.
        asi como nuevas posiciones random de los GUs
        """
        #reseteamos la figura y creamos visualmente de nuevo al robot en una posicion random.
        self.reset_env = True
        #random position robot...

        random_new_pos_robots = np.random.random(size=(3,len(obj_drone_list)))
        random_new_pos_robots[0,:] = random_new_pos_robots[0,:] * self.boundaries[2]
        random_new_pos_robots[1,:] = random_new_pos_robots[1,:] * self.boundaries[3]
        random_new_pos_robots[2,:] = random_new_pos_robots[2,:] * 2 * np.pi

        #update robot poses in environment objects...
        self.poses = random_new_pos_robots

        #update robot poses in mobileagent objects...
        mobileagent.MobileAgent.set_drone_poses(obj_drone_list,self.poses)

        #create drones in new position..     y reseteamos listas con data de drones y gus   
        self.generateVisualRobots(rc,rc_color)

        #initialize position robots...        
        self.step_v2(obj_gus_list,obj_drone_list,True)

        #random position gus...
        random_new_pos_gus = np.random.random(size=(3,len(obj_gus_list)))
        random_new_pos_gus[0,:] = random_new_pos_gus[0,:] * self.boundaries[2]
        random_new_pos_gus[1,:] = random_new_pos_gus[1,:] * self.boundaries[3]
        random_new_pos_gus[2,:] = random_new_pos_gus[2,:] * 2 * np.pi

        #update gu poses in environment objects...
        self.poses_gus = random_new_pos_gus

        #update gu poses in gu objects...
        gu.GroundUser.set_gu_poses(obj_gus_list,self.poses_gus,obj_drone_list)
        

        #reset gus in new position.. and velocities
        self.velocities_gus = np.zeros([2,len(obj_gus_list)])    
 
        self.resetGUs(ObjGuList = obj_gus_list,Pose = random_new_pos_gus, Radius = graph_rad, FaceColor = list_color_gus,
                           PlotDataRate = True)
        
        #initialize position GU...        
        self.step_v2(obj_gus_list,obj_drone_list)

        self.reset_env = False


    def isTerminalState(self,obj_drone_list,obj_gu_list, tolerance = 0.01):
        """
        esta funcion evaluara si el estado actual del ambiente es terminal.
        por el momento el estado terminal sera si el drone navega fuera de los limites del espacio 2d o
        algun gu navega fuera del ambiente
        """
        for obj in obj_drone_list + obj_gu_list:
            if (obj.pose[0] > self.boundaries[2] or obj.pose[0] < self.boundaries[0]) or (obj.pose[1] > self.boundaries[3]  or
                            obj.pose[1] < self.boundaries[1]):
                return True
        return False



    def getState(self,obj_drone_list,obj_gu_list,max_gu_data):
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
        env_state[0,0] = obj_drone_list[0].dict_gu["Gu_0"]["DistanceToDrone"]
        env_state[0,1] = obj_drone_list[0].dict_gu["Gu_0"]["Connection"]
        env_state[0,2] = np.interp(obj_gu_list[0].transmission_rate, [0,max_gu_data],[0, 1])
        env_state[0,3] = obj_drone_list[0].dict_gu["Gu_1"]["DistanceToDrone"]
        env_state[0,4] = obj_drone_list[0].dict_gu["Gu_1"]["Connection"]
        env_state[0,5] = np.interp(obj_gu_list[1].transmission_rate, [0,max_gu_data],[0, 1])

       

        #tuple_gu_pos = tuple((gu.pose for gu in obj_gu_list))
        #tuple_gu_dic = (obj_drone_list[0].dict_gu)
        
        return env_state#(obj_drone_list[0].pose, tuple_gu_pos, tuple_gu_dic)
        

    def getAreaDimensions(self):
        #retornamos punto origen, width, height
        return self.boundaries
    
    def showDrones(self,**args):
        """
        esta funcion permite generar los circulos que emulan el comportamiento rc de cada drone
        """
        i = args["Index"]
        center_rc = args["Pose"][:2,i] #index i is for robot i

        self.robots_rc_circles.append(patches.Circle(center_rc,args["Rc"],fill = False,
        facecolor = args["FaceColor"]))

        self.axes.add_patch(self.robots_rc_circles[i])


    def createDrones(self):
        obj_list_drones = []
        for i in range(self.number_of_robots):
            obj_list_drones.append(mobileagent.MobileAgent("Drone_" + str(i),self.poses[:,i]))
        return obj_list_drones
    
    def updateGUDataRate(self,gu_index,trans_data_rate ):
        """
        actualizaremos valor del data rate en ambiente...
        """
        self.gu_tb_data[gu_index].set_text(str(np.round(trans_data_rate,2)))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def createGUs(self, **args):
        """
        esta funcion permitira crear los objetos gus, asi como mostrarlos en el ambiente.
        """
        obj_list_gus = []

        num_gus = args["Pose"].shape[1]
        self.poses_gus = np.zeros([3,num_gus])
        self.velocities_gus = np.zeros([2,num_gus])

        for i in range(num_gus):
            obj_list_gus.append(gu.GroundUser("Gu_" + str(i),args["Pose"][:,i]))
            self.poses_gus[:,i] = args["Pose"][:,i]
            #obj_list_gus[i].setDistanceToDrone(args["PoseDrone"])

            self.showGUs(Index = i, Radius =  args["Radius"],FaceColor = args["FaceColor"],
                         PlotDataRate = args["PlotDataRate"] )

        if self.show_figure: #sim visual
            plt.ion()
            plt.show()

        return obj_list_gus

    def resetGUs(self, **args):
        """
        esta funcion dibujara a los gus despues de resetear sus posiciones y data rates

        """
        num_gus = self.poses_gus.shape[1]
        for i in range(num_gus):
            #reseteamos data rates
            args["ObjGuList"][i].transmission_rate = 0.0

            self.showGUs(Index = i, Radius =  args["Radius"],FaceColor = args["FaceColor"],
                         PlotDataRate = args["PlotDataRate"] )

        if self.show_figure: #sim visual
            plt.ion()
            plt.show()


    def showGUs(self, **args):
        """
        funciones necesarias para crear tanto al gu, como su texto de data transmission
        """
        i = args["Index"]
        self.gu_patches.append(patches.Circle(self.poses_gus[:2,i],args["Radius"],fill = True,
        facecolor = args["FaceColor"][i]))
        self.axes.add_patch(self.gu_patches[i])

        #create data rate gu textbox, update: since we are creating/reseting gus, then trans rate starts at -1
        if args["PlotDataRate"]:
            self.gu_tb_data.append(self.axes.text(self.poses_gus[0,i],self.poses_gus[1,i],
                                                          0.0))


        
    



