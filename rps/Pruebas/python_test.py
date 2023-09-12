"""
pickle test: dump a graphic in file every t iterations
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
fig = plt.figure()
size = 10
#plt.plot(np.random.randint(0,10,size=(size,)))
#plt.xlabel("Number of episodes")
#plt.ylabel("Mean Reward")

# save whole figure 
trained_path = "C:/Users/kevin/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/MCC/Tesis/Project Drone 2D/Drone-2D/rps/Pruebas/"
model_name = "updating_graph_test"
l = [10,15,20,20]
for _ in range(size):
    with open(trained_path + model_name + ".txt","a+") as f:
        f.write(str(np.random.randint(0,10)) + ",")
    #pickle.dump("hola como estas", open(trained_path + model_name +".txt", "wb"))

#open graphic

data =  open(trained_path + model_name + ".txt", "r") 
dt = (data.read()).split(",")



#repeat...

fig = plt.figure()
size = 100
plt.plot(np.random.randint(0,10,size=(size,)))
plt.xlabel("Number of episodes")
plt.ylabel("Mean Reward")

pickle.dump(fig, open(trained_path + model_name +".pickle", "wb"))

#open graphic
fig_handle = pickle.load(open(trained_path + model_name+".pickle",'rb'))
fig_handle.show()

