import numpy as np
import rps.robotarium as robotarium

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rps.Modules.gu as gu

class graphGU:
    def __init__(self) -> None:
        """
        esta clase contendra la informacion de las caracteristicas intrinsicas del GU (posicion, transmision data,etc.),
        ademas 
        """
        pass

class environment(robotarium.Robotarium):
    """
    esta superclase extendiende la funcionalidad del robotarium para poder trabajar con un ambiente integrado con GUs,
    así como sus respectivas clases
    """
    def __init__(self,boundaries, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):
        super().__init__(boundaries,number_of_robots, show_figure, sim_in_real_time, initial_conditions)

        #visualization GUs
        self.gu_patches = []

    def getAreaDimensions(self):
        #retornamos punto origen, width, height
        return self.boundaries
        
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




