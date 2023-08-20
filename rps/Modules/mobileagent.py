"""
Creamos el modelo con las caracteristicas del agente en si (el vehiculo). incluiremos:
posicion agente, mecanismos para cambiar su posicion

ademas, modelaremos el radio con el cual podra detectar usuarios en tierra, asi como guardar informacion valiosa
de ellos. el entrenamiento de la red actor-critic, el cual sera con un sistema central, utilizara esta informacion para
obtener el estado global del sistema y poder entrenar el control de este agente
"""

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
    def __init__(self,dict_gu,):
        pass

    def echoRadio(self):
        """
        adquirir info gus en el rango rc del drone. incluir logica identificacion cambio de registros de posicion.
        
        """
        pass


class MobileAgent:
    def __init__(self,pos,rc):
        pass

    def updatePosition(self,direction,distance):
        """
        actualizaremos posicion del agente, dependiendo de la accion seleccionada por el algortimo de control
        si el radio detecta modificaciones en su registro de posiciones de los ususarios. el drone accionara su
        mecanismo para actualizar su posicion y poder seguir con el cambio.
        """
        pass


    def getPosition(self):
        pass

