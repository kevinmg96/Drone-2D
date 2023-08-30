import numpy as np
import rps.robotarium as robotarium

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rps.Modules.gu as gu
import rps.Modules.mobileagent as mobileagent


class environment(robotarium.Robotarium):
    """
    esta superclase extendiende la funcionalidad del robotarium para poder trabajar con un ambiente integrado con GUs,
    así como sus respectivas clases
    """
    def __init__(self,boundaries, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):
        super().__init__(boundaries,number_of_robots, show_figure, sim_in_real_time, initial_conditions)



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

        #create drones in new position..        
        self.generateVisualRobots()

        self.createDrones(Drone = obj_drone_list,Pose = random_new_pos_robots, Rc = rc,
                          FaceColor = rc_color)

        #initialize position robots...        
        self.step_v2(obj_gus_list,obj_drone_list,True)

        #random position gus...
        random_new_pos_gus = np.random.random(size=(3,len(obj_gus_list)))
        random_new_pos_gus[0,:] = random_new_pos_gus[0,:] * self.boundaries[2]
        random_new_pos_gus[1,:] = random_new_pos_gus[1,:] * self.boundaries[3]
        random_new_pos_gus[2,:] = random_new_pos_gus[2,:] * 2 * np.pi

        #initialize new position gus..

        self.createGUs(GU = obj_gus_list,Pose = random_new_pos_gus, Radius = graph_rad, FaceColor = list_color_gus,
                           PlotDataRate = True)
        
        #initialize position GU...        
        self.step_v2(obj_gus_list,obj_drone_list)

        self.reset_env = False


    def isTerminalState(self,obj_drone_list, tolerance = 0.01):
        """
        esta funcion evaluara si el estado actual del ambiente es terminal.
        por el momento el estado terminal sera si el drone navega fuera de los limites del espacio 2d
        """
        if np.abs(obj_drone_list[0].pose[0] - self.boundaries[2]) < tolerance or \
        np.abs(obj_drone_list[0].pose[1] - self.boundaries[3]) < 0.01:
            return True
        return False



    def getState(self,obj_drone_list,obj_gu_list):
        """
        esta funcion nos permitira obtener el estado global del sistema. 
        (Pose drone, Pose GUs, isGUcovered, GU tranmission rate
        """       

        tuple_gu_pos = tuple((gu.pose for gu in obj_gu_list))
        tuple_gu_dic = (obj_drone_list[0].dict_gu)
        
        return (obj_drone_list[0].pose, tuple_gu_pos, tuple_gu_dic)
        

    def getAreaDimensions(self):
        #retornamos punto origen, width, height
        return self.boundaries
    
    def createDrones(self,**args):
        """
        funcion que permite crear el radio de conectividad de los drones, asi como el objeto con la data especifica del drone
        
        Actualizacion Fecha: 29/08/2023-> modifique el codigo para que funcione con el reset environment
        """
        if not self.reset_env:
            obj_list_drones = []

        for i in range(self.number_of_robots):
            #create robot object
            if self.reset_env:
                drone_pose_curr = args["Pose"][:,i]
                if len(drone_pose_curr.shape) == 1: #reshape
                    drone_pose_curr =  drone_pose_curr.reshape(-1,1)                   
                mobileagent.MobileAgent.set_drone_poses([args["Drone"][i],],drone_pose_curr)                
            else:
                obj_list_drones.append(mobileagent.MobileAgent("Drone_" + str(i),args["Pose"][:,i]))


            #create drone rc circle
            if self.show_figure: #sim visual

                center_rc = args["Pose"][:2,i]

                self.robots_rc_circles.append(patches.Circle(center_rc,args["Rc"],fill = False,
                facecolor = args["FaceColor"]))
                self.axes.add_patch(self.robots_rc_circles[i])



        if self.show_figure: #sim visual
            plt.ion()
            plt.show()

        if not self.reset_env:
            return obj_list_drones

    
    def updateGUDataRate(self,gu_index,trans_data_rate ):
        """
        actualizaremos valor del data rate en ambiente...
        """
        self.gu_tb_data[gu_index].set_text(str(np.round(trans_data_rate,2)))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        
    def createGUs(self,**args):
        """
        esta funcion permitira crear a los GUs (si la sim es visual, tmb los creara en el terreno) 
        que estarán viviendo en el ambiente
        Actualizacion Fecha : 29/08/2023 ->
        actualice la funcion para que funcione para resetear la posicion de los gus.
        """
        if  not self.reset_env: #primera creacion de los gus...
            obj_list_gus =  []


        num_gus = args["Pose"].shape[1]
        self.poses_gus = np.zeros([3,num_gus])

        for i in range(num_gus):
            #create intrinsic object gu, if env reset we do not create the intrinsic object, instead update gus pose
            if self.reset_env: #actualizamos posicion de cada gu...
                gu_pose_curr = args["Pose"][:,i]
                if len(gu_pose_curr.shape) == 1: #reshape
                    gu_pose_curr =  gu_pose_curr.reshape(-1,1) 
                gu.GroundUser.set_gu_poses([args["GU"][i]],gu_pose_curr)

            else: #nuevo gu...    
                obj_list_gus.append(gu.GroundUser("GU_" + str(i),args["Pose"][:,i]))

            self.poses_gus[:,i] = args["Pose"][:,i]

            #create gu patch
            if self.show_figure: #sim visual
                self.gu_patches.append(patches.Circle(self.poses_gus[:2,i],args["Radius"],fill = True,
                facecolor = args["FaceColor"][i]))
                self.axes.add_patch(self.gu_patches[i])

                #create data rate gu textbox, update: since we are creating/reseting gus, then trans rate starts at -1
                if args["PlotDataRate"]:
                    self.gu_tb_data.append(self.axes.text(self.poses_gus[0,i],self.poses_gus[1,i],
                                                          -1))

        if self.show_figure: #sim visual
            plt.ion()
            plt.show()
        else:
            pass
            #self.figure.set_visible(False)
            #plt.draw()

        #set velocities starting from equilibrium
        self.velocities_gus = np.zeros((2, num_gus))

        if not self.reset_env:
            return obj_list_gus




