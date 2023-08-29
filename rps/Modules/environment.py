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

        #visualization GUs
        self.gu_patches = []
        self.gu_tb_data = []
        self.robots_rc_circles = []

    def resetEnv(self):
        """
        resetereamos la posicion del robot inicial a una random.
        asi como nuevas posiciones random de los GUs
        """
        self.reset_env = True
        #random position robot...
        self.generateVisualRobots()

        #random position gus...


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
        """
        obj_list_drones = []

        for i in range(self.number_of_robots):
            #create robot object
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
        """
        obj_list_gus =  []

        num_gus = len(args["Pose"])
        self.poses_gus = np.zeros([3,num_gus])

        for i in range(num_gus):
            #create intrinsic object gu
            obj_list_gus.append(gu.GroundUser("GU_" + str(i),args["Pose"][i]))
            self.poses_gus[:,i] = obj_list_gus[i].pose

            #create gu patch
            if self.show_figure: #sim visual
                self.gu_patches.append(patches.Circle(obj_list_gus[i].pose[:2],args["Radius"],fill = True,
                facecolor = args["FaceColor"][i]))
                self.axes.add_patch(self.gu_patches[i])

            #create data rate gu textbox
                if args["PlotDataRate"]:
                    self.gu_tb_data.append(self.axes.text(self.poses_gus[0,i],self.poses_gus[1,i],
                                                          np.round(obj_list_gus[i].transmission_rate,2)))

        if self.show_figure: #sim visual
            plt.ion()
            plt.show()
        else:
            pass
            #self.figure.set_visible(False)
            #plt.draw()

        #set velocities starting from equilibrium
        self.velocities_gus = np.zeros((2, num_gus))


        return obj_list_gus




