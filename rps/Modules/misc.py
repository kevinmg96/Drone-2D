"""
libreria de funciones 
"""
import numpy as np
import scipy.stats as stats

def euclideanDistance(point_1,point_2):

    p1_z = 0
    p2_z = 0
    
    if len(point_1) > 2: #puntos 3d
        p1_z = point_1[-1]
        p2_z = point_2[-1]

    return np.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2 + (p2_z - p1_z) ** 2)




def poissonChoice(mag_max,mean = 3,samp_size = 100000,bool_debug = False):

    samp_size = 10000
    sample = np.array(stats.poisson.rvs(mu = mean, size = samp_size))
    dic = {}
    for s in sample:
        if not s in dic:
            dic[s] = 1
        else:
            dic[s] += 1

    for key in dic.keys():
        dic[key] = dic[key] / samp_size

    #sort keys
    list_keys = list(dic.keys()) 
    list_keys.sort()

    #construct array of probability distribution
    list_p_values = []
    for key in list_keys:
        list_p_values.append(dic[key])
    
    arr_p_values = np.array(list_p_values)

    #create random different positions of same size as the number of different random variables from poisson distribution
    pos_next_values = np.arange(len(list_keys)) * (mag_max/ (len(list_keys)-1))

    if bool_debug:
        print("random variables: {}, count: {}".format(dic.keys(),dic.values()))
        print("sorted keys: {}".format(list_keys))
        print("probability values: {}".format(list_p_values))
        print("possible next positions: {}".format(pos_next_values))

    
    return np.random.choice(pos_next_values,size = 1, p = arr_p_values)[0]


def randomChoice(mag_max):
    return np.random.random(size = 1)[0] * mag_max



def moveObjNextStep():
    """
    funcion que aleatoriamente determinara si el objeto requiere cambiar posicion o tranmision de datos
    """
    return np.random.randint(0,2,1)[0].astype(bool)

def computeNextPosition(next_mag,pos,bool_debug = False):
    """
    calculamos la nueva posicion del robot o gu, ingresando el rango maximo a poder desplazarse.
    angulo permitido : 0 a 2 pi
    actualizacion: modifique la funcion para que funcionara con una distribucion de las nuevas posiciones
    """
    if bool_debug:
        print("ingrese funcion")

    
    rand_ang = np.random.random(size = (1,1))[0] * 2 * np.pi 
    if bool_debug:
        print("ang rand: {}".format(rand_ang))

    new_pos = np.zeros([2,])

    new_pos[0] = pos[0] + np.cos(rand_ang) * next_mag

    new_pos[1] = pos[1] + np.sin(rand_ang) * next_mag

    if bool_debug:
        print("new pos : {}".format(new_pos))

    return new_pos

