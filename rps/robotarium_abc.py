import time
import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rps.Modules.mobileagent as mobileagent
import rps.Modules.gu as gu

import rps.utilities.misc as misc

# RobotariumABC: This is an interface for the Robotarium class that
# ensures the simulator and the robots match up properly.  

# THIS FILE SHOULD NEVER BE MODIFIED OR SUBMITTED!

class RobotariumABC(ABC):

    def __init__(self,boundaries,initial_conditions,show_figure=True, sim_in_real_time=True,**kwargs):

        #Check user input types
        assert isinstance(initial_conditions,np.ndarray), "The initial conditions array argument (initial_conditions) provided to create the Robotarium object must be a numpy ndarray. Recieved type %r." % type(initial_conditions).__name__
        assert isinstance(show_figure,bool), "The display figure window argument (show_figure) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(show_figure).__name__
        assert isinstance(sim_in_real_time,bool), "The simulation running at 0.033s per loop (sim_real_time) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(show_figure).__name__
        
        
        self.number_of_robots = initial_conditions.shape[1]
        self.show_figure = show_figure
        self.initial_conditions = initial_conditions

        # Boundary stuff -> lower left point / width / height
        self.boundaries = boundaries#[-1.6, -1, 3.2, 2]

        self.file_path = None
        self.current_file_size = 0

        # Constants
        self.time_step = 0.033
        self.robot_diameter = 0.11
        self.wheel_radius = 0.016
        self.base_length = 0.105
        self.max_linear_velocity = 0.2
        self.max_angular_velocity = 2*(self.wheel_radius/self.robot_diameter)*(self.max_linear_velocity/self.wheel_radius)
        self.max_wheel_velocity = self.max_linear_velocity/self.wheel_radius

        self.robot_radius = self.robot_diameter/2
        self.robot_length = 0.095
        self.robot_width = 0.09

        self.collision_offset = 0.025 # May want to increase this
        self.collision_diameter = 0.135

        self.reset_env = False

        #create the object containing the data for all the drones
        self.createDrones(Rc = kwargs["Rc"], FaceColor = kwargs["FaceColor"])

        #generate drones visual effects
        self.generateVisualRobots()

        #create gus visual effects and obj
        self.createGUs(PoseGu = kwargs["PoseGu"], GuRadius = kwargs["GuRadius"], GuColorList = kwargs["GuColorList"],
                       PlotDataRate = kwargs["PlotDataRate"], MaxGuDist = kwargs["MaxGuDist"],MaxGuData = kwargs["MaxGuData"],
            StepGuData = kwargs["StepGuData"])
        
        self.left_led_commands = []
        self.right_led_commands = []

    #------------------------------------ FUNCTIONS RELATED TO DRONES -----------------------------------------------#

    def showDrones(self,**args):
        """
        esta funcion permite generar los circulos que emulan el comportamiento rc de cada drone
        """
        i = args["Index"]
        center_rc = args["Pose"][:2,i] #index i is for robot i

        self.robots_rc_circles.append(patches.Circle(center_rc,args["Rc"],fill = False,
        facecolor = args["FaceColor"]))

        self.axes.add_patch(self.robots_rc_circles[i])

    def createDrones(self,**kwargs):
        self.obj_drones = mobileagent.MobileAgent(["Drone_" + str(i) for i in range(self.number_of_robots)],
                                        self.initial_conditions,kwargs["Rc"],kwargs["FaceColor"])
        
    #------------------------------------ FUNCTIONS RELATED TO DRONES -----------------------------------------------#

    #----------------------------------- FUNCTIONS RELATED TO GUS ---------------------------------------------------#

    def updateGUDataRate(self,gu_index,trans_data_rate ):
        """
        actualizaremos valor del data rate en ambiente...
        """
        self.gu_tb_data[gu_index].set_text(str(np.round(trans_data_rate,2)))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


    def resetGUs(self):
        """
        esta funcion dibujara a los gus despues de resetear sus posiciones y data rates

        """
        #reset visualization GUs lists
        self.gu_patches = []
        self.gu_tb_data = []

        num_gus = self.obj_gus.poses.shape[1]
        #reseteamos data rates
        self.obj_gus.transmission_rate[:] = 0.0
        for i in range(num_gus):

            if self.show_figure:
                self.showGUs(Index = i)

        if self.show_figure: #sim visual
            plt.ion()
            plt.show()


    def showGUs(self, **args):
        """
        funciones necesarias para crear tanto al gu, como su texto de data transmission
        """
        i = args["Index"]
        self.gu_patches.append(patches.Circle(self.obj_gus.poses[:2,i],self.obj_gus.visual_radius,fill = True,
        facecolor = self.obj_gus.gus_list_colors[i]))
        self.axes.add_patch(self.gu_patches[i])

        #create data rate gu textbox, update: since we are creating/reseting gus, then trans rate starts at 0.0
        if self.obj_gus.plot_data_rate:
            self.gu_tb_data.append(self.axes.text(self.obj_gus.poses[0,i],self.obj_gus.poses[1,i],
                                                          0.0))


    def createGUs(self, **args):
        """
        esta funcion permitira crear los objetos gus, asi como mostrarlos en el ambiente.
        """
        #visualization GUs
        self.gu_patches = []
        self.gu_tb_data = []

        num_gus = args["PoseGu"].shape[1]
        self.obj_gus = gu.GroundUser(["Gu_" + str(i) for i in range(num_gus)],args["PoseGu"],args["GuRadius"],
                                     args["GuColorList"],args["PlotDataRate"],args["MaxGuDist"],args["MaxGuData"],
                                     args["StepGuData"])

        for i in range(num_gus):
            #obj_list_gus[i].setDistanceToDrone(args["PoseDrone"])
            if self.show_figure:
                self.showGUs(Index = i)

        if self.show_figure: #sim visual
            plt.ion()
            plt.show()

    
    #----------------------------------- FUNCTIONS RELATED TO GUS ---------------------------------------------------#

    def generateVisualRobots(self):
        """
        esta funcion permitira iniciar o resetear el ambiente con los objetos que dan forma visual al robot
        """

        # Visualization
        #self.figure = []
        #self.axes = []
        self.left_led_patches = []
        self.right_led_patches = []
        self.chassis_patches = []
        self.right_wheel_patches = []
        self.left_wheel_patches = []
        self.base_patches = []
        self.robots_rc_circles = []
                
        
        if(self.show_figure):
            if self.reset_env:
                #self.figure.clear()
                self.axes.clear()
                #self.obj_drones.reset_drones_velocities()
            else:
                self.figure, self.axes = plt.subplots()
                #print("tipo objeto figure : {}".format(type(self.figure)))
                #print("tipo objeto axes : {}".format(type(self.axes)))
                self.axes.set_axis_off()
            
            for i in range(self.number_of_robots):
                p = patches.Rectangle((self.obj_drones.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]+math.pi/2), np.sin(self.obj_drones.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2)))), self.robot_length, self.robot_width, (self.obj_drones.poses[2, i] + math.pi/4) * 180/math.pi, facecolor='#FFD700', edgecolor='k')

                rled = patches.Circle(self.obj_drones.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i]))+0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2)))),
                                       self.robot_length/2/5, fill=False)
                lled = patches.Circle(self.obj_drones.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i]))+\
                                        0.015*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2)))),\
                                       self.robot_length/2/5, fill=False)
                rw = patches.Circle(self.obj_drones.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]+math.pi/2), np.sin(self.obj_drones.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')
                lw = patches.Circle(self.obj_drones.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]-math.pi/2), np.sin(self.obj_drones.poses[2, i]-math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')

                self.chassis_patches.append(p)
                self.left_led_patches.append(lled)
                self.right_led_patches.append(rled)
                self.right_wheel_patches.append(rw)
                self.left_wheel_patches.append(lw)
   

                self.axes.add_patch(rw)
                self.axes.add_patch(lw)
                self.axes.add_patch(p)
                self.axes.add_patch(lled)
                self.axes.add_patch(rled)


                #generate the Rc circle for each robot...
                self.showDrones(Index = i,Pose = self.obj_drones.poses,Rc = self.obj_drones.rc,FaceColor = self.obj_drones.rc_color)

            # Draw arena
            self.boundary_patch = self.axes.add_patch(patches.Rectangle(self.boundaries[:2], self.boundaries[2], self.boundaries[3], fill=False))

            self.axes.set_xlim(self.boundaries[0]-0.1, self.boundaries[0]+self.boundaries[2]+0.1)
            self.axes.set_ylim(self.boundaries[1]-0.1, self.boundaries[1]+self.boundaries[3]+0.1)

            plt.ion()
            plt.show()

            plt.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=1.03, wspace=0, hspace=0)
        else:
            pass
            #self.figure.set_visible(False)
            #plt.draw()

    def set_velocities(self,velocities,bool_robot = False):
                    
        # Threshold linear velocities
        idxs = np.where(np.abs(velocities[0, :]) > self.max_linear_velocity)
        velocities[0, idxs] = self.max_linear_velocity*np.sign(velocities[0, idxs])

        # Threshold angular velocities
        idxs = np.where(np.abs(velocities[1, :]) > self.max_angular_velocity)
        velocities[1, idxs] = self.max_angular_velocity*np.sign(velocities[1, idxs])

        if bool_robot:#update drones velocities
            self.obj_drones.velocities = velocities 
        else:#update gus velocities
            self.obj_gus.velocities = velocities

   
    #Protected Functions
    def _threshold(self, dxu):
        dxdd = self._uni_to_diff(dxu)

        to_thresh = np.absolute(dxdd) > self.max_wheel_velocity
        dxdd[to_thresh] = self.max_wheel_velocity*np.sign(dxdd[to_thresh])

        dxu = self._diff_to_uni(dxdd)

    def _uni_to_diff(self, dxu):
        r = self.wheel_radius
        l = self.base_length
        dxdd = np.vstack((1/(2*r)*(2*dxu[0,:]-l*dxu[1,:]),1/(2*r)*(2*dxu[0,:]+l*dxu[1,:])))

        return dxdd

    def _diff_to_uni(self, dxdd):
        r = self.wheel_radius
        l = self.base_length
        dxu = np.vstack((r/(2)*(dxdd[0,:]+dxdd[1,:]),r/l*(dxdd[1,:]-dxdd[0,:])))

        return dxu

    def _validate(self, errors = {}):
        # This is meant to be called on every iteration of step.
        # Checks to make sure robots are operating within the bounds of reality.

        p = self.poses
        b = self.boundaries
        N = self.number_of_robots

        for i in range(N):
            x = p[0,i]
            y = p[1,i]

            if(x < b[0] or x > (b[0] + b[2]) or y < b[1] or y > (b[1] + b[3])):
                    if "boundary" in errors:
                        if i in errors["boundary"]:
                            errors["boundary"][i] += 1
                        else:
                            errors["boundary"][i] = 1
                            
                    else:
                        errors["boundary"] = {i: 1}
                        errors["boundary_string"] = "iteration(s) robots were outside the boundaries."

        for j in range(N-1):
            for k in range(j+1,N):
                first_position = p[:2, j] + self.collision_offset*np.array([np.cos(p[2,j]), np.sin(p[2, j])])
                second_position = p[:2, k] + self.collision_offset*np.array([np.cos(p[2,k]), np.sin(p[2, k])])
                if(np.linalg.norm(first_position - second_position) <= (self.collision_diameter)):
                # if (np.linalg.norm(p[:2,j]-p[:2,k]) <= self.robot_diameter):
                    if "collision" in errors:
                        if j in errors["collision"]:
                            errors["collision"][j] += 1
                        else:
                            errors["collision"][j] = 1
                        # if k == N:
                        if k in errors["collision"]:
                            errors["collision"][k] += 1
                        else:
                            errors["collision"][k] = 1
                    else:
                        errors["collision"]= {j: 1}
                        errors["collision"][k] = 1
                        # if k == N:

                        errors["collision_string"] = "iteration(s) where robots collided."

        dxdd = self._uni_to_diff(self.velocities)
        exceeding = np.absolute(dxdd) > self.max_wheel_velocity
        if(np.any(exceeding)):
            if "actuator" in errors:
                errors["actuator"] += 1
            else:
                errors["actuator"] = 1
                errors["actuator_string"] = "iteration(s) where the actuator limits were exceeded."

        return errors

