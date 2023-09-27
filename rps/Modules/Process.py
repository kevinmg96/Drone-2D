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

    def guProcess(self,r,unicycle_position_controller,pose_eval,bool_debug = False):
        
        num_gus = r.obj_gus.poses.shape[1]
        goal_points_gus = np.zeros([3,num_gus])

        # define x initially for  gus
        x_gus = r.obj_gus.poses       

        #unpausing event
        self.pause_mobility_event.set()

        while True:#not self.stop_mobility_event.is_set():
            #get goal points randomly for robots and gus
            
            if bool_debug:
                start = time.time()

            #indicate if gu needs updating position.
            for i in range(num_gus):
                #get next magnitude displacement probability distribution
                next_disp_distribution = [misc.randomChoice(r.obj_gus.max_gu_dist),misc.poissonChoice(r.obj_gus.max_gu_dist),
                                        misc.gaussianChoice(r.obj_gus.max_gu_dist)]
                p_distribution = [0.35,0.5,0.15]
                next_dist_gu = np.random.choice(next_disp_distribution,p=p_distribution) 
                              
                if bool_debug:
                    print("GU : {}, new direction ? : {}".format(r.obj_gus.ids[i],r.obj_gus.bool_new_direction[i] ))

                if r.obj_gus.bool_new_direction[i]:

                    #set new random position to displace gu             
                    next_possible_pos_gu,gu_direction = misc.computeNextPosition(next_dist_gu,x_gus[:2,i])
                    next_possible_pos_gu = next_possible_pos_gu.reshape(-1,1)
                    r.obj_gus.curr_direction[i] = gu_direction 
                else:
                    #continue pursuing same direction...
                    next_possible_pos_gu,_ = misc.computeNextPosition(next_dist_gu,x_gus[:2,i],
                    r.obj_gus.curr_direction[i],False)
                    next_possible_pos_gu = next_possible_pos_gu.reshape(-1,1)

                terminal_state = r.isTerminalState(next_possible_pos_gu)

                if bool_debug:
                    print("GU : {}, next possible position : {}".format(r.obj_gus.ids[i],next_possible_pos_gu ))
                    print("is terminal state ?: {}".format(terminal_state))
                
                if terminal_state: #keep current position
                    goal_points_gus[:,i] = x_gus[:,i]
                else: #take next gu position
                    goal_points_gus[:2,i] = next_possible_pos_gu[:,0]               
                
                if bool_debug:
                    print("GU : {}, final next position: {}".format(r.obj_gus.ids[i],goal_points_gus[:2,i]))

                #set bool_new_direction for next possible gus position based on a distribution of probabilities
                bool_new_direction_distribution = [misc.randomChoice(1.0),misc.poissonChoice(1.0,5),
                misc.gaussianChoice(1.0)]
                p_distribution = [0.1,0.8,0.1]
                bool_new_direction_sel = np.random.choice(bool_new_direction_distribution,p=p_distribution)

                r.obj_gus.bool_new_direction[i] = np.where(bool_new_direction_sel < 0.6,False,True)

                #actualizamos estatus data tranmission y rate de los gus
                r.obj_gus.setTransmissionRate(r.obj_gus.max_gu_data,False)#misc.randomChoice(r.obj_gus.max_gu_data),False)
                
                #actualizamos data transmission value en ambiente
                if r.show_figure:
                    r.updateGUDataRate(i,r.obj_gus.transmission_rate[i])


            if r.show_figure: #if debugging results, use single integrator to drive the dynamics of the gus
                #actualizamos posiciones de los gus...
                while (np.size(pose_eval(x_gus, goal_points_gus,position_error = 0.05, rotation_error=300)) != num_gus):
                    #get pose gus
                    x_gus = r.obj_gus.poses 

                    # Create single-integrator control inputs for gus
                    dxu_gus = unicycle_position_controller(x_gus, goal_points_gus[:2,:])

                    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
                    r.set_velocities(dxu_gus)

                    # Iterate the simulation
                    r.step_v2()
            else: # if training, the next drone positions will be automatically used to update gus pose
                r.obj_gus.poses = goal_points_gus
            
            if bool_debug:           
                end = time.time()
                print("iteration delta time : {} s".format(end - start))


            #pause event
            self.pause_mobility_event.wait()

            if self.stop_mobility_event.is_set(): #stop event is activated, break loop
                break


        