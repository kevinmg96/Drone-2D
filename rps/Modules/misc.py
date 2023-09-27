"""
libreria de funciones 
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
def euclideanDistance(point_1,point_2):

    p1_z = 0
    p2_z = 0
    
    if len(point_1) > 2: #puntos 3d
        p1_z = point_1[-1]
        p2_z = point_2[-1]

    return np.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2 + (p2_z - p1_z) ** 2)


def gaussianChoice(mag_max,std_dev = 0.1, samp_size = 100000,bool_debug = False):

    #create the gaussian probability distribution of size samp_size
    x = np.linspace(0,1,samp_size)
    prob_x = stats.norm.pdf(x,loc = 0.5, scale = std_dev)
    prob_x = prob_x / np.sum(prob_x)

    if bool_debug:
        #debugearemos la prob dist
        plt.plot(prob_x)
        plt.show()

    #create an array from 0 to mag_max, same size as x
    pos_y_value = np.linspace(0,mag_max,samp_size)



    return np.random.choice(pos_y_value,p=prob_x)

def poissonChoice(mag_max,mean = 3,samp_size = 100000,bool_debug = False):


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

def computeNextPosition(next_mag,pos,direction = 2 * np.pi,bool_pos_gu = True,bool_debug = False):
    """
    calculamos la nueva posicion del robot o gu, ingresando el rango maximo a poder desplazarse.
    angulo permitido : 0 a 2 pi
    actualizacion: modifique la funcion para que funcionara con una distribucion de las nuevas posiciones
    """
    if bool_debug:
        print("ingrese funcion")

    if bool_pos_gu: #la distancia que estamos calculando sera un gu, calculamos angulo random
        dir = np.random.random(size = (1,1))[0] * direction
    else: #la distancia es para un drone, ingresamos la direccion
        dir = direction
    if bool_debug:
        print("ang rand: {}".format(dir))

    new_pos = np.zeros([2,])

    new_pos[0] = pos[0] + np.cos(dir) * next_mag

    new_pos[1] = pos[1] + np.sin(dir) * next_mag

    if bool_debug:
        print("new pos : {}".format(new_pos))

    return new_pos,dir

if __name__ == "__main__":
    max_distance = 5.0
    dist_distributions = [randomChoice(max_distance), poissonChoice(max_distance),gaussianChoice(max_distance) ]