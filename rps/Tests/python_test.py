"""
prueba stop/resume thread
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def poissonProbDist(mag_max,mean = 3,samp_size = 100000,bool_debug = False):

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



poissonProbDist(0.25,10000,True)   




