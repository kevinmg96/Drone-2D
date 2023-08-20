import numpy as np
import rps.robotarium as robotarium

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class environment(robotarium.Robotarium):
    def __init__(self,boundaries, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([])):
        super().__init__(boundaries,number_of_robots, show_figure, sim_in_real_time, initial_conditions)

    def getAreaDimensions(self):
        #retornamos punto origen, width, height
        return self.boundaries
    
    def createGU(self,gu):
        """
        esta funcion permitira crear al GU en el robotarium
        """
        gu_graphic = patches.Circle(gu.pos,gu.graph_rad,fill = True, facecolor = "r")
        self.axes.add_patch(gu_graphic)

        plt.ion()
        plt.show()


