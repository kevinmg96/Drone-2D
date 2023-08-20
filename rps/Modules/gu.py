"""
modelo que emulara el comportamiento de los GU.
guardara su posicion,
tendremos un modelo random para detectar si el gu desea moverse de posicion. asi como su desplazamiento sera random.
tendremos otro modelo de la tasa de tranmision, tendremos un numero random para detectar si este gu desea compartir data,
asi como un numero random para detectar su magnitud. Mantendremos esos parametros por un intervalo de tiempo. despues
resetearemos las specs

tasa de tranmision lo incluire como un textbox en el grafo.
"""

class GroundUser:
    def __init__(self,id,pos,graph_rad):
        self.id = id
        self.pos = pos
        self.graph_rad = graph_rad

