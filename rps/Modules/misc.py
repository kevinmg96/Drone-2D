"""
libreria de funciones 
"""
import numpy as np

def moveObjNextStep():
    """
    funcion que aleatoriamente determinara si el objeto requiere cambiar posicion o tranmision de datos
    """
    return np.random.randint(0,2,1)[0].astype(bool)

def computeNextPosition(mag_max,pos,bool_debug = False):
    """
    calculamos la nueva posicion del robot o gu, ingresando el rango maximo a poder desplazarse.
    angulo permitido : 0 a 2 pi
    """
    if bool_debug:
        print("ingrese funcion")
    rand_mag = np.random.random(size = (1,1))[0] * mag_max

    if bool_debug:
        print("mag rand : {}".format(rand_mag))

    rand_ang = np.random.random(size = (1,1))[0] * 2 * np.pi 
    if bool_debug:
        print("ang rand: {}".format(rand_ang))

    new_pos = np.zeros([2,])

    new_pos[0] = pos[0] + np.cos(rand_ang) * rand_mag

    new_pos[1] = pos[1] + np.sin(rand_ang) * rand_mag

    if bool_debug:
        print("new pos : {}".format(new_pos))

    return new_pos

