import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rps.robotarium_abc import *
import rps.Modules.gu as gu
import rps.Modules.mobileagent as mobileagent

# Robotarium This object provides routines to interface with the Robotarium.
#
# THIS CLASS SHOULD NEVER BE MODIFIED OR SUBMITTED

class Robotarium(RobotariumABC):

        def __init__(self,boundaries,initial_conditions,show_figure=True, sim_in_real_time = True,**kwargs):
            super().__init__(boundaries,initial_conditions, show_figure, sim_in_real_time,
            Rc = kwargs["Rc"],FaceColor = kwargs["FaceColor"], PoseGu = kwargs["PoseGu"], GuRadius = kwargs["GuRadius"], 
            GuColorList = kwargs["GuColorList"],
            PlotDataRate = kwargs["PlotDataRate"], MaxGuDist = kwargs["MaxGuDist"],MaxGuData = kwargs["MaxGuData"],
            StepGuData = kwargs["StepGuData"])

            #Initialize some rendering variables
            self.previous_render_time = time.time()
            self.sim_in_real_time = sim_in_real_time

        
        def step_v2(self,bool_robot = False,bool_debug = False):
            """Increments the simulation by updating the dynamics.
            actualizacion fecha: 21/08/2023.
            esta funcion actualizara la pose de robots y gus
            """
 
            # Update dynamics of agents robots and gus
            if bool_robot:
                self.obj_drones.set_drone_poses(self.time_step)
                
            else:
                self.obj_gus.set_gu_poses(self.time_step)
            
            # Update graphics
            if(self.show_figure):
                if(self.sim_in_real_time):
                    t = time.time()
                    while(t - self.previous_render_time < self.time_step):
                        t=time.time()
                    self.previous_render_time = t

                if bool_robot:

                    for i in range(self.number_of_robots):
                        
                        self.chassis_patches[i].xy = self.obj_drones.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]+math.pi/2), np.sin(self.obj_drones.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i])))
  
                        self.chassis_patches[i].angle = (self.obj_drones.poses[2, i] - math.pi/2) * 180/math.pi

                        self.right_wheel_patches[i].center = self.obj_drones.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]+math.pi/2), np.sin(self.obj_drones.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i])))
                        self.right_wheel_patches[i].orientation = self.obj_drones.poses[2, i] + math.pi/4

                        self.left_wheel_patches[i].center = self.obj_drones.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]-math.pi/2), np.sin(self.obj_drones.poses[2, i]-math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.obj_drones.poses[2, i]+math.pi/2), np.cos(self.obj_drones.poses[2, i]+math.pi/2))) + self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i])))
                        self.left_wheel_patches[i].orientation = self.obj_drones.poses[2,i] + math.pi/4
                        
                        self.right_led_patches[i].center = self.obj_drones.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2,i]), np.sin(self.obj_drones.poses[2,i])))-\
                                        0.04*np.array((-np.sin(self.obj_drones.poses[2, i]), np.cos(self.obj_drones.poses[2, i]))) + self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i])))
                        self.left_led_patches[i].center = self.obj_drones.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2,i]), np.sin(self.obj_drones.poses[2,i])))-\
                                        0.015*np.array((-np.sin(self.obj_drones.poses[2, i]), np.cos(self.obj_drones.poses[2, i]))) + self.robot_length/2*np.array((np.cos(self.obj_drones.poses[2, i]), np.sin(self.obj_drones.poses[2, i])))
                        # self.base_patches[i].center = self.poses[:2, i]

                        #update center mobileagent rc
                        self.robots_rc_circles[i].center = self.obj_drones.poses[:2, i]
                else:
                    for i in range(self.obj_gus.poses.shape[1]):
                        self.gu_patches[i].center = self.obj_gus.poses[:2, i]

                        #update center gu transmission rate textbox
                        self.gu_tb_data[i].set_position(self.gu_patches[i].center)

                        if bool_debug:
                            print("gu index : {}, circle center: {}, textbox center: {}".format(i,self.gu_patches[i].center,
                                                                self.gu_tb_data[i].get_position()))


                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()



