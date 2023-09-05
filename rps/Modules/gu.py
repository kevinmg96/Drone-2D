"""
modelo que emulara el comportamiento de los GU.
guardara su posicion,
tendremos un modelo random para detectar si el gu desea moverse de posicion. asi como su desplazamiento sera random.
tendremos otro modelo de la tasa de tranmision, tendremos un numero random para detectar si este gu desea compartir data,
asi como un numero random para detectar su magnitud. Mantendremos esos parametros por un intervalo de tiempo. despues
resetearemos las specs

tasa de tranmision lo incluire como un textbox en el grafo.
"""
import numpy as np
import rps.Modules.misc as misc




class GroundUser:
    def __init__(self,id,pose):
        self.id = id
        self.pose = pose #vector : x,y,angle
        self.transmission_rate = 0.0
        self.is_gu_transmiting = False

    def setDistanceToDrone(self,drone_pose):
        """
        esta funcion seteara una variable del objeto, indicando la distancia qeu existe entre el gu i y drone
        """
        self.distance_to_drone = misc.euclideanDistance(self.pose[:2],drone_pose[:2])

    def setTransmissionRate(self,g_x,gu_trans_max,step_gu_data,bool_debug = False):
        """
        1.- variable para indicar que este gu desea tranmitir data
        2.- variable random para cambiar el estado del proceso. decidir si continuar mandando data o parar.
        3.- si trans data == 0.0, entonces setearemos la tranmission rate random, si no entonces agregaremos o decrementaremos
        una cantidad x al valor actual
        actualizacion: seleccionar la tasa a comenzar a transmitir data dependera de la distribucion de probabilidad deseada
        """
        self.is_gu_transmiting = np.where(misc.poissonChoice(1.0,5) < 0.6,True,False)#misc.moveObjNextStep()

        if self.is_gu_transmiting: #gu quiere continuar o empezar a transmitir data
            if np.abs(self.transmission_rate) < 0.001: #gu no ha empezado a transmitir ninguna data
                self.transmission_rate = g_x 
            else: #data esta siendo transmitida, agregaremos o decremetnaremos valor por un step
                if misc.moveObjNextStep(): #actualizaremos data rate
                    temp_rate = self.transmission_rate + np.where(misc.moveObjNextStep(),1,-1) * step_gu_data
                    if not(temp_rate > gu_trans_max or temp_rate < 0.0):
                        self.transmission_rate = temp_rate
        else: #gu quiere terminar de transmitir
            self.transmission_rate = 0.0

        if bool_debug:
            print("gu : {}, transmission status: {}, data rate: {}".format(self.id,self.is_gu_transmiting,self.transmission_rate))



    @staticmethod
    def get_gu_poses(obj_list_gus):
        arr_pose_gus = np.zeros([3,len(obj_list_gus)])

        for i,gu in enumerate(obj_list_gus):
            arr_pose_gus[:3,i] = gu.pose[:3]

        return arr_pose_gus
    
    @staticmethod
    def set_gu_poses(obj_list_gus,arr_pose_gus,obj_drone_list = None):

        for i,gu in enumerate(obj_list_gus):
            gu.pose = arr_pose_gus[:,i]
            #if not obj_drone_list == None: #update distance to drone...
                #obj_list_gus[i].setDistanceToDrone(obj_drone_list[0].pose)

        