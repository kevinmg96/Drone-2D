"""
modelo que emulara el comportamiento de los GU.
guardara su posicion,
tendremos un modelo random para detectar si el gu desea moverse de posicion. asi como su desplazamiento sera random.
tendremos otro modelo de la tasa de tranmision, tendremos un numero random para detectar si este gu desea compartir data,
asi como un numero random para detectar su magnitud. Mantendremos esos parametros por un intervalo de tiempo. despues
resetearemos las specs

tasa de tranmision lo incluire como un textbox en el grafo.

update: el objeto creado con la clase contendra la data de todso los gus desplegados...
"""
import numpy as np
import rps.Modules.misc as misc




class GroundUser:
    def __init__(self,ids,poses,visual_radius,gus_list_colors,plot_data_rate,max_gu_dist,
                 max_gu_data,step_gu_data):
        self.ids = ids
        self.poses = poses
        self.velocities = np.zeros([2,len(self.ids)]) 
        self.transmission_rate = np.zeros([len(self.ids),])
        self.is_gu_transmiting = [False for _ in range(len(self.ids))]
        self.visual_radius = visual_radius #this is the visual radius of the gu (circle)
        self.gus_list_colors  = gus_list_colors#colors of the filled circle of each gu
        self.plot_data_rate = plot_data_rate
        self.max_gu_dist = max_gu_dist
        self.max_gu_data = max_gu_data #transmission rate
        self.step_gu_data = step_gu_data
        

    def setDistanceToDrone(self,drone_pose):
        """
        esta funcion seteara una variable del objeto, indicando la distancia qeu existe entre el gu i y drone
        """
        self.distance_to_drone = misc.euclideanDistance(self.pose[:2],drone_pose[:2])

    def setTransmissionRate(self,g_x,bool_debug = False):
        """
        1.- variable para indicar que este gu desea tranmitir data
        2.- variable random para cambiar el estado del proceso. decidir si continuar mandando data o parar.
        3.- si trans data == 0.0, entonces setearemos la tranmission rate random, si no entonces agregaremos o decrementaremos
        una cantidad x al valor actual
        actualizacion: seleccionar la tasa a comenzar a transmitir data dependera de la distribucion de probabilidad deseada
        """

        for i in range(len(self.ids)):
            self.is_gu_transmiting[i] = np.where(misc.poissonChoice(1.0,5) < 0.6,True,False)#misc.moveObjNextStep()

            if self.is_gu_transmiting[i]: #gu quiere continuar o empezar a transmitir data
                if np.abs(self.transmission_rate[i]) < 0.001: #gu no ha empezado a transmitir ninguna data
                    self.transmission_rate[i] = misc.randomChoice(g_x) 
                else: #data esta siendo transmitida, agregaremos o decremetnaremos valor por un step
                    if misc.moveObjNextStep(): #actualizaremos data rate
                        temp_rate = self.transmission_rate[i] + np.where(misc.moveObjNextStep(),1,-1) * self.step_gu_data
                        if not(temp_rate > self.max_gu_data or temp_rate < 0.0):
                            self.transmission_rate[i] = temp_rate
            else: #gu quiere terminar de transmitir
                self.transmission_rate[i] = 0.0

            if bool_debug:
                print("gu : {}, transmission status: {}, data rate: {}".format(self.ids[i],
                                                self.is_gu_transmiting[i],self.transmission_rate[i]))

    def reset_gus_velocities(self):
        self.velocities = np.zeros([2,self.poses.shape[1]])

    @staticmethod
    def get_gu_poses(obj_list_gus):
        arr_pose_gus = np.zeros([3,len(obj_list_gus)])

        for i,gu in enumerate(obj_list_gus):
            arr_pose_gus[:3,i] = gu.pose[:3]

        return arr_pose_gus
    
    def set_gu_poses(self,time_step):
        self.poses[0, :] = self.poses[0, :] + time_step*np.cos(self.poses[2,:])*self.velocities[0, :]
        self.poses[1, :] = self.poses[1, :] + time_step*np.sin(self.poses[2,:])*self.velocities[0, :]
        self.poses[2, :] = self.poses[2, :] + time_step*self.velocities[1, :]            

        self.poses[2, :] = np.arctan2(np.sin(self.poses[2, :]), np.cos(self.poses[2, :]))

        