"""
Creamos el modelo con las caracteristicas del agente en si (el vehiculo). incluiremos:
posicion agente, mecanismos para cambiar su posicion

ademas, modelaremos el radio con el cual podra detectar usuarios en tierra, asi como guardar informacion valiosa
de ellos. el entrenamiento de la red actor-critic, el cual sera con un sistema central, utilizara esta informacion para
obtener el estado global del sistema y poder entrenar el control de este agente
UPDATE : fecha : 12/09/2023 actualice este modulo para almacenar la data pose echo para todos los drones,
es decir un agente contendra la info de todos los drones..
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
    def __init__(self,ids,poses):
        self.ids = ids
        self.poses = poses
        self.velocities = np.zeros([2,self.poses.shape[1]])
        self.dict_gu = [{} for _ in range(self.poses.shape[1])]

    def echoRadio(self,obj_gus,bool_debug = False):
        """
        funcion que permite adquirir el estatus de la cobertura de los gus
        
        Fecha actualizacion : 31/08/2023 -> el radio transmisor del drone tendra la capacidad de detectar las posiciones
        de los gus desplegados (por el momento es centralizado el sistema)
        """
        for i in range(self.poses.shape[1]):
            for j in range(obj_gus.poses.shape[1]):

                delta = misc.euclideanDistance(self.poses[:2,i],obj_gus.poses[:2,j])

                if obj_gus.ids[j] in self.dict_gu[i]:
                    self.dict_gu[i][obj_gus.ids[j]]["DataRate"] = obj_gus.transmission_rate[j]
                    self.dict_gu[i][obj_gus.ids[j]]["Position"] = obj_gus.poses[:2,j]
                    self.dict_gu[i][obj_gus.ids[j]]["DistanceToDrone"] = delta
                else:
                    self.dict_gu[i][obj_gus.ids[j]] = {"DataRate" : obj_gus.transmission_rate[i],
                                            "Position" : obj_gus.poses[:2,j], "DistanceToDrone" : delta }
                
                if delta < self.rc:
                    self.dict_gu[i][obj_gus.ids[j]]["Connection"] = True
                else:
                    self.dict_gu[i][obj_gus.ids[j]]["Connection"] = False

                """
                if delta < rc: #communication link between drone i and gu k
                    if gu.id in self.dict_gu:
                        self.dict_gu[gu.id]["Connection"] = True
                        self.dict_gu[gu.id]["DataRate"] = gu.transmission_rate
                        self.dict_gu[gu.id]["Position"] = gu.pose[:2]
                        self.dict_gu[gu.id]["DistanceToDrone"] = delta
                    else: #new object to be stored
                        self.dict_gu[gu.id] = {"Connection" : True, "DataRate" : gu.transmission_rate,
                                            "Position" : gu.pose[:2] }
                else:
                    #gu k fuera de rango de drone i
                    if gu.id in self.dict_gu: #drone i cuenta con info previa de gu k
                        self.dict_gu[gu.id]["Connection"] = False
                    else: #agregaremos el gu al dorne i aunque el drone no sea capaz de detectarlo (modelo centralizado)
                        self.dict_gu[gu.id] = {"Connection" : False}
                """
            if bool_debug:
                print("Drone: {}, gu dict: {}".format(self.ids[i], self.dict_gu[i]))




class MobileAgent(Radio):
    def __init__(self,ids,poses,rc,rc_color):
        super().__init__(ids,poses)

        #visual characteristics drones
        self.rc = rc
        self.rc_color = rc_color

    def get_drone_poses(self):
        return self.poses
    
    def set_drone_poses(self,time_step):
        self.poses[0, :] = self.poses[0, :] + time_step*np.cos(self.poses[2,:])*self.velocities[0, :]
        self.poses[1, :] = self.poses[1, :] + time_step*np.sin(self.poses[2,:])*self.velocities[0, :]
        self.poses[2, :] = self.poses[2, :] + time_step*self.velocities[1, :]

        # Ensure angles are wrapped
        self.poses[2, :] = np.arctan2(np.sin(self.poses[2, :]), np.cos(self.poses[2, :]))

    
    def reset_drones_velocities(self):
        self.velocities = np.zeros([2,self.poses.shape[1]])



