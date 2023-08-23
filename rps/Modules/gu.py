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
        self.transmission_rate = -1
        self.is_gu_transmiting = False

    def setTransmissionRate(self,max_gu_data,step_gu_data):
        """
        1.- variable para indicar que este gu desea tranmitir data
        2.- variable random para cambiar el estado del proceso. decidir si continuar mandando data o parar.
        3.- si trans data == -1, entonces setearemos la tranmission rate random, si no entonces agregaremos o decrementaremos
        una cantidad x al valor actual
        """
        self.is_gu_transmiting = misc.moveObjNextStep()

        if self.is_gu_transmiting: #gu quiere continuar o empezar a transmitir data
            if self.transmission_rate == -1: #gu no ha empezado a transmitir ninguna data
                self.transmission_rate = np.random.random(size = (1,1))[0] * max_gu_data
            else: #data esta siendo transmitida, agregaremos o decremetnaremos valor por un step
                self.transmission_rate + np.where(misc.moveObjNextStep(),1,-1) * step_gu_data
        else: #gu quiere terminar de transmitir
            self.transmission_rate = -1



    @staticmethod
    def get_gu_poses(obj_list_gus):
        arr_pose_gus = np.zeros([3,len(obj_list_gus)])

        for i,gu in enumerate(obj_list_gus):
            arr_pose_gus[:3,i] = gu.pose[:3]

        return arr_pose_gus
    
    @staticmethod
    def set_gu_poses(obj_list_gus,arr_pose_gus):

        for i,gu in enumerate(obj_list_gus):
            gu.pose = arr_pose_gus[:,i]

        