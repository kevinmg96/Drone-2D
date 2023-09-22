"""
este modulo contendra todas las clases y funciones que permitan desarrollar el entrenamiento por DQN de nuestro agente.
"""

#--------------------------------------------- TENSORFLOW CPU --------------------------------------------------------#

"""
import keras
from keras.layers import Dense
from keras.models import Sequential,load_model
from keras.optimizers import RMSprop 
from tensorflow import gather_nd
from keras.losses import mean_squared_error

import keras
"""
#-------------------------------------------- TENSORFLOW CPU .----------------------------------------------------------------#

#--------------------------------------------- TENSORFLOW GPU --------------------------------------------------------#

import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error



#-------------------------------------------- TENSORFLOW GPU .----------------------------------------------------------------#


from collections import namedtuple, deque
import random
import numpy as np


from rps.utilities.misc import *

import glob
import os
import re

Transition = namedtuple('Transition',
                        ('state', 'action','reward','next_state', "is_terminal"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        #self.memory.append(args[0])
        self.memory.append(Transition(*args))

    def drop_left(self):
        self.memory.popleft()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def __getsize__(self):
        return self.memory.maxlen

def soft_update(target_model, primary_model, tau):
    target_weights = target_model.get_weights()
    primary_weights = primary_model.get_weights()
    
    updated_weights = []
    for i in range(len(target_weights)):
        updated_weights.append(tau * primary_weights[i] + (1 - tau) * target_weights[i])
    
    target_model.set_weights(updated_weights)

#-------------------------------- MODEL BACKUP FUNCTIONS ---------------------------------------------------------#
    
def save_model(model,path):
    """
    esta funcion permitira guardar el modelo q(state,action) value network en la ruta especificada
    """
    model.save(path)

def save_pretrained_model(model,folder_path,filename):

    list_of_files = glob.glob(folder_path + "*.keras")
    if list_of_files == []: #folder is empty
        full_path = folder_path + filename + "--1.keras"
    else: #find latest file and increase its version
        latest_model = max(list_of_files, key=os.path.getctime)

        version_model=re.findall(filename + '--\d+', latest_model)[0]
        version_model_index = version_model.find("--")
        full_path = folder_path + filename + "--" + str(int(version_model[version_model_index + 2:]) + 1) + ".keras"
    model.save(full_path)

def load_info_data(full_data_path):
    data_file =  open(full_data_path, "r") 
    data_stream = data_file.read()
    data_split = data_stream.split(",")
    data_split.pop()
    data_array = np.array(data_split,dtype=float)
    return data_array

def plot_rewards(data):
    fig = plt.figure()
    plt.scatter(np.arange(start =1 , stop = len(data) + 1), data)
    plt.xlabel("Number of episodes")
    plt.ylabel("Mean Reward")
    plt.show()

#-------------------------------- MODEL BACKUP FUNCTIONS ---------------------------------------------------------#

class DQNAgent:
    def __init__(self,state_dimension,action_space, mem_capacity,gamma,epsilon,num_episodes, batch_size,
                timeslot_train_iter_max = 1000,pretrained_iter_saver = 500,target_network_update_interval = 300,model_full_path = ""):

        self.state_dimension = state_dimension
        self.action_space = action_space
        self.memoryBuffer = ReplayMemory(mem_capacity)
        self.counter_train_timeslot = 0 #este contador permitira salir del inner loop training, si el sistema no encuentra un 
        #estado terminal
        self.timeslot_train_iter_max = timeslot_train_iter_max

        #salvar el modelo preentrado cada cierto numero de iteraciones. parametro ingresado a la clase
        self.counter_iter = 0
        self.pretrained_iter_saver = pretrained_iter_saver

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.exploration_proba_decay = 0.005
        self.tau = 0.005 # soft update: este valor lentamente reducira los pesos del target network e ira acercando mas
        #los pesos hacia los valores de q network
        self.counter_update_target_network  = 0#este counter incrementará durante cada timestep
        self.target_network_update_interval = target_network_update_interval 

        #mean de las recompensas por episodio
        self.meanRewardsEpisode = "" # change methodology to save it in string format[]

        #create q network NN model
        self.loss_function = self.my_loss_fn
        if model_full_path == "": #create q network from scratch
            self.q_network = self.createNetwork()
        else: #load pretrained model
            self.load_keras_model(model_full_path)


        #create target network NN model
        self.target_network = self.createNetwork()
        self.target_network.set_weights(self.q_network.get_weights())


        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend=[]
    
    def load_keras_model(self,model_full_path):
        self.q_network = load_model(model_full_path,compile=False)
        self.q_network.compile(optimizer = RMSprop(), loss = self.loss_function, metrics = ['accuracy'])


    def update_exploration_probability(self,bool_debug = False):
        self.epsilon = self.epsilon * np.exp(-self.exploration_proba_decay)

        if bool_debug:
            print(self.epsilon) 

    def my_loss_fn(self,y_true, y_pred):
        #y_true -> target q value network prediction on new_state max(q_target_network(next_state)) plus reward ->>> y_i
        # y_pred -> prediction of the main q_network(input_network_data_set, output_network) -> q state action value
        s1,s2=y_true.shape
        #print("shape y_true : {}".format((s1,s2)))
        #print("matrix y_true:{}".format(y_true))

        s3,s4=y_pred.shape
        #print("shape y_pred : {}".format((s3,s4)))
        #print("matrix y_pred:{}".format(y_pred))
        # this matrix defines indices of a set of entries that we want to 
        # extract from y_true and y_pred
        # s2=2
        # s1=self.batchReplayBufferSize
        indices=np.zeros(shape=(s1,2))
        indices[:,0]=np.arange(s1)
        indices[:,1]=self.actionsAppend

        loss = mean_squared_error(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        #print(loss)
        return loss 

    #create neural network
    def createNetwork(self):
        """
        esta funcion permitira crear la estructura de las redes DNN para este agente DQN
        """
        model=Sequential()
        model.add(Dense(128,input_dim = self.state_dimension,activation='relu')) #añadir aqui state dimmension
        model.add(Dense(56,activation='relu'))
        model.add(Dense(self.action_space.shape[0],activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer = RMSprop(), loss = self.loss_function, metrics = ['accuracy'])
        return model




    def trainingEpisodes(self,env,obj_process_mob_trans_gu,folder_pretrained_model = "",
                        model_name = "",info_data_file = "",bool_debug = False,**args):
        # gu process: en esta seccion del entrenamiento, ejecutaremos las acciones permitidas a los gus.
        #ejecutamos accion del drone utilizando la heuristica epsilon-greedy
        #ejecutamos accion drone en simulador y obtenemos reward de accion , asi como nuevo estado
        # almacenamos transition en buffer
        #si el replay buffer ya tiene su capacidad maxima llena, entonces procedemos con el entrenamiento de la network
        for i in range(self.num_episodes):
            rewards_per_episode = []

            if bool_debug:
                print("Simulating episode {}".format(i))

            #reseteamos el ambiente, al inicio de cada episodio...           
            env.resetEnv()
           
            #actualizamos los registros del drone...
            env.obj_drones.echoRadio(env.obj_gus)

            #get initial state
            current_state = env.getState()
            
            is_next_state_terminal = False

            while not is_next_state_terminal: #mientras no hemos llegado a un estado terminal, continuar iterando avanzando en el episodio
                self.counter_train_timeslot += 1
                if self.counter_train_timeslot >self.timeslot_train_iter_max: #salimos del inner loop. cambiamos a otro ep
                    break
                

                #seleccionamos accion para agente de acuerdo a e greedy strategy
                index_action, action = self.selectAction(current_state)

                #ejecutamos accion del DQN agent (desplazamos al drone...), retornamos new_state,reward,is_terminal_state
                next_state,reward,is_next_state_terminal = env.stepEnv(action,at_pose,args["PositionController"])
                rewards_per_episode.append(reward)


                #store the transition tuple...
                self.memoryBuffer.push(current_state,(index_action, action),reward,next_state,is_next_state_terminal)
                

                #train q_network...
                if self.memoryBuffer.__len__() > self.batch_size: #si tenemos el minimo de transiciones necesarias
                    #para poder crear el batchbuffer, procedemos al entrenamiento de la red q network
                    self.trainNetwork()

                    #save pretrained model and save a text file with mean episode rewards
                    if not folder_pretrained_model == "":
                        self.counter_iter += 1
                        if self.counter_iter > self.pretrained_iter_saver:
                            save_pretrained_model(self.q_network,folder_pretrained_model,model_name)
                            self.counter_iter = 0

                            #mean episode rewards save
                            with open(folder_pretrained_model + info_data_file + ".txt","a+") as f:
                                f.write(self.meanRewardsEpisode)
                                self.meanRewardsEpisode = ""

                if is_next_state_terminal: #if next state is terminal, then finish the training episode and restart the process...
                    #update exploration probability...
                    self.update_exploration_probability()
                    break

                current_state = next_state
                
                # ejecutamos acciones de los gus
                obj_process_mob_trans_gu.guProcess(env,args["PositionController"],at_pose,False)

                #actualizamos los registros del drone...
                env.obj_drones.echoRadio(env.obj_gus)   

                #si el nuevo estado del ambiente es terminal, un gu se desplazo fuera de los limites, terminamos este episodio
                if env.isTerminalState():
                    break            
                
            if bool_debug:
                print("mean rewards : {},episode : {}, num. iterations: {}".format(np.mean(rewards_per_episode), i,
                                                                                   self.counter_train_timeslot))        
            self.meanRewardsEpisode += str(np.round(np.mean(rewards_per_episode),3)) + ","

            self.counter_train_timeslot = 0

    def selectAction(self,state):
        """
        esta funcion permite calcular la nueva accion a tomar por el agente, de acuerdo a la epsilon-greedy policy
        """
        if np.random.random() < self.epsilon: # si rand num menor a epsilon ejecutamos accion greedy
            index_action =np.random.choice(np.arange(self.action_space.shape[0]))
            return index_action,self.action_space[index_action,:]
        else:
            # calculamos el valor q(state,action) utilizando main network
            q_values =self.q_network.predict(state,verbose = 0) 
            return np.argmax(q_values),self.action_space[np.argmax(q_values),:]

    def trainNetwork(self,bool_debug = False):             
 
        # sample a batch from the replay buffer
        randomSampleBatch = self.memoryBuffer.sample(self.batch_size)
            
        # here we form current state batch 
        # and next state batch
        # they are used as inputs for prediction
        currentStateBatch=np.zeros(shape=(self.batch_size,self.state_dimension))
        nextStateBatch=np.zeros(shape=(self.batch_size,self.state_dimension))            
        # this will enumerate the tuple entries of the randomSampleBatch
        # index will loop through the number of tuples
        for index,tupleS in enumerate(randomSampleBatch):
            # first entry of the tuple is the current state
            currentStateBatch[index,:]=tupleS[0]
            # fourth entry of the tuple is the next state
            nextStateBatch[index,:]=tupleS[3]
             
        # here, use the target network to predict Q-values 
        QnextStateTargetNetwork=self.target_network.predict(nextStateBatch,verbose = 0)
        # here, use the main network to predict Q-values 
        QcurrentStateMainNetwork=self.q_network.predict(currentStateBatch,verbose = 0)
             
        # now, we form batches for training
        # input for training
        inputNetwork=currentStateBatch
        # output for training
        outputNetwork=np.zeros(shape=(self.batch_size,self.action_space.shape[0]))
             
        # this list will contain the actions that are selected from the batch 
        # this list is used in my_loss_fn to define the loss-function
        self.actionsAppend=[]    
      
        for index,(currentState,action,reward,nextState,terminated) in enumerate(randomSampleBatch):
                 
            # if the next state is the terminal state
            if terminated:
                y=reward                  
            # if the next state if not the terminal state    
            else:
                y=reward+self.gamma*np.max(QnextStateTargetNetwork[index])
                 
            # this is necessary for defining the cost function index action
            self.actionsAppend.append(action[0])
                 
            # this actually does not matter since we do not use all the entries in the cost function
            outputNetwork[index]=QcurrentStateMainNetwork[index]
            # this is what matters
            outputNetwork[index,action[0]]=y

        if bool_debug:
            print("actionAppend : {}".format(self.actionsAppend))
            print("input_network : {}".format(inputNetwork))
            print("output network :{}".format(outputNetwork))
            print("QcurrentStateMainNetwork :{}".format(QcurrentStateMainNetwork))

        # here, we train the network
        self.q_network.fit(inputNetwork,outputNetwork,batch_size = self.batch_size, verbose=0,epochs=100)  

        #

        #soft update of the target network parameters to get into the q network parameters  
         # θ′ ← τ θ + (1 −τ )θ′ 
        #soft_update(self.target_network,self.q_network,self.tau)

        #update the target networks if counter == update interval
        self.counter_update_target_network += 1
        if self.counter_update_target_network == self.target_network_update_interval:
            self.target_network.set_weights(self.q_network.get_weights())
            self.counter_update_target_network = 0


             
       

if __name__ == "__main__":
    #testearemos lso argumentos de la clase DQN
    state_dimension = 6
    n_transitions = 100
    action_space = np.random.randint(0,5,size=(4,2))
    
    #
    batch_size = 10
    agent = DQNAgent(state_dimension,action_space,n_transitions,0.1,0.6,100,batch_size)


    for i in range(n_transitions):
        current_state = np.random.random(size=(1,state_dimension))
        index_action = np.random.choice(np.arange(action_space.shape[0]))
        action = (index_action,action_space[index_action])
        reward = np.random.random()
        next_state = np.random.random(size=(1,state_dimension))
        is_terminal = np.random.randint(0,2,dtype=bool)

        agent.memoryBuffer.push(current_state,action,reward,next_state,is_terminal)

    agent.trainNetwork()