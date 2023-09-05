"""
este modulo contendra las clases de los procesos de la simulacion
"""
import time
import rps.Modules.misc as misc
import numpy as np
import rps.Modules.gu as gu
from multiprocessing import Event

class ProcesGuMobility:
    """
    esta clase contendra el proceso que govierne la dinamica de los gus, así como su requerimientos de calidad de servicio.
    Nota: los procesos derivados de esta clase controlan a los M gus desplegados. de forma secuencial.
    """
    def __init__(self) -> None:
        self.stop_mobility_event = Event()
        self.pause_mobility_event = Event()

    def setStopProcess(self):
        self.stop_mobility_event.set()

    def clearStopProcess(self):
        self.stop_mobility_event.clear()

    def pauseProcess(self,bool = False):
        if bool: #pause event
            self.pause_mobility_event.clear()
        else: #resume
            self.pause_mobility_event.set()

    def guProcess(self,r,obj_list_gus,obj_drone_list,max_gu_dist,max_transmission_rate,step_trans_data,
                  unicycle_position_controller,pose_eval,bool_debug = False):
        num_gus = len(obj_list_gus)
        goal_points_gus = np.zeros([3,num_gus])

        # define x initially for  gus
        x_gus = gu.GroundUser.get_gu_poses(obj_list_gus)
        #r.step_v2(obj_list_gus,obj_drone_list)

        #unpausing event
        self.pause_mobility_event.set()

        while True:#not self.stop_mobility_event.is_set():
            #get goal points randomly for robots and gus
            
            if bool_debug:
                start = time.time()

            #indicate if gu needs updating position.
            for i in range(num_gus):
                if np.where(misc.poissonChoice(1.0,5) < 0.5,True,False):#misc.moveObjNextStep():
                    #update gu position
                    goal_points_gus[:2,i] = misc.computeNextPosition(misc.poissonChoice(max_gu_dist),x_gus[:2,i]) 
                else:
                    #keep current position
                    goal_points_gus[:2,i] = x_gus[:2,i]

                if bool_debug:
                    print("GU : {}, new position: {}".format(obj_list_gus[i].id,goal_points_gus[:2,i]))

                #actualizamos estatus data tranmission y rate de los gus
                obj_list_gus[i].setTransmissionRate(misc.randomChoice(max_transmission_rate),max_transmission_rate,
                                                    step_trans_data,False)
                
                #actualizamos data transmission value en ambiente
                if r.show_figure:
                    r.updateGUDataRate(i,obj_list_gus[i].transmission_rate)



            #actualizamos posiciones de los gus...
            while (np.size(pose_eval(x_gus, goal_points_gus,position_error = 0.05, rotation_error=100)) != num_gus):
                #get pose gus
                x_gus = gu.GroundUser.get_gu_poses(obj_list_gus)

                # Create single-integrator control inputs for gus
                dxu_gus = unicycle_position_controller(x_gus, goal_points_gus[:2,:])

                # Set the velocities by mapping the single-integrator inputs to unciycle inputs
                r.set_velocities(np.arange(num_gus), dxu_gus)

                # Iterate the simulation
                r.step_v2(obj_list_gus,obj_drone_list)
            
            if bool_debug:           
                end = time.time()
                print("iteration delta time : {} s".format(end - start))


            #pause event
            self.pause_mobility_event.wait()

            if self.stop_mobility_event.is_set(): #stop event is activated, break loop
                break


        