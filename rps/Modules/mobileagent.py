"""
Creamos el modelo con las caracteristicas del agente en si (el vehiculo). incluiremos:
posicion agente, mecanismos para cambiar su posicion

ademas, modelaremos el radio con el cual podra detectar usuarios en tierra, asi como guardar informacion valiosa
de ellos. el entrenamiento de la red actor-critic, el cual sera con un sistema central, utilizara esta informacion para
obtener el estado global del sistema y poder entrenar el control de este agente
"""
import numpy as np
import rps.Modules.misc as misc
class Radio:
    """
    modelo del radio transmisor. permite la funcionalidad de establecer vinculos de comunicacion con usuarios en tierra
    dentro de las limitaciones del radio transmisor. 
    conservara en memoria registros de la ubicacion de los agentes en su rango de cobertura y si cuenta con conexion o no.
    detectaremos un cambio de informacion y activaremos un registro. si este registro es true, el agente sabra 
    que hubo un cambio en el estado y accionara el mecanismo de escoger una nueva posicion (trabajo del drone)

    como los registros se encuentran en un espacio en memoria del radio transmisor. entonces existe la posibilidad
    de que exista un booleano capaz de activarse cuadno los valores de uno cambien.
    """
    def __init__(self,id,pose):
        self.id = id
        self.pose = pose
        self.dict_gu = {}

    def echoRadio(self,obj_list_gus,rc,bool_debug = False):
        """
        funcion que permite adquirir el estatus de la cobertura de los gus
        
        """
        for gu in obj_list_gus:
            delta = misc.euclideanDistance(self.pose[:2],gu.pose[:2])

            if delta < rc: #communication link between drone i and gu k
                if gu.id in self.dict_gu:
                    self.dict_gu[gu.id]["Connection"] = True
                    self.dict_gu[gu.id]["DataRate"] = gu.transmission_rate
                else: #new object to be stored
                    self.dict_gu[gu.id] = {"Connection" : True, "DataRate" : gu.transmission_rate}
            else:
                #gu k fuera de rango de drone i
                if gu.id in self.dict_gu: #drone i cuenta con info previa de gu k
                    self.dict_gu[gu.id]["Connection"] = False

        if bool_debug:
            print("Drone: {}, gu dict: {}".format(self.id, self.dict_gu))




class MobileAgent(Radio):
    def __init__(self,id,pose):
        super().__init__(id,pose)
        self.id = id
        self.pose = pose


    @staticmethod
    def get_drone_poses(obj_list_drones):
        arr_pose_drones = np.zeros([3,len(obj_list_drones)])

        for i,drone in enumerate(obj_list_drones):
            arr_pose_drones[:3,i] = drone.pose[:3]

        return arr_pose_drones
    
    @staticmethod
    def set_drone_poses(obj_list_drones,arr_pose_drones):

        for i,drone in enumerate(obj_list_drones):
            drone.pose = arr_pose_drones[:,i]

    def updatePosition(self,direction,distance):
        """
        actualizaremos posicion del agente, dependiendo de la accion seleccionada por el algortimo de control
        si el radio detecta modificaciones en su registro de posiciones de los ususarios. el drone accionara su
        mecanismo para actualizar su posicion y poder seguir con el cambio.
        """
        pass


    def getPosition(self):
        pass

