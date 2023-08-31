"""
este modulo contendra todas las clases y funciones que permitan desarrollar el entrenamiento por DQN de nuestro agente.
"""
from collections import namedtuple, deque
import random
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop 
from tensorflow import gather_nd
from keras.losses import mean_squared_error
import numpy as np

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

memory =ReplayMemory(100)
t = Transition(10,15,20,30,40)
memory.push(10,15,20,30,40)
memory.push(20,15,20,30,40)
print(memory.memory.maxlen)
print(memory.__len__())
    
class DQNAgent:
    def __init__(self,state_dimension,action_space, mem_capacity,gamma,epsilon,num_episodes, batch_size ) -> None:

        self.state_dimension = state_dimension
        self.action_space = action_space
        self.memoryBuffer = ReplayMemory(mem_capacity)

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.exploration_proba_decay = 0.005

        #suma de las recompensas por episodio
        self.sumRewardsEpisode = []

        #create q network NN model
        self.q_network = self.createNetwork()

        #create target network NN model
        self.target_network = self.createNetwork()
        self.target_network.set_weights(self.q_network.get_weights())


        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend=[]
    
    def getReward(self):
        """
        esta funcion implementara la ecuacion para obtener la recompensa por la accion ejecutada del drone.
        R = conec_1 * data_rate_1 + conec_2 * data_rate_2
        """
        pass

    def update_exploration_probability(self,bool_debug = False):
        self.epsilon = self.epsilon * np.exp(-self.exploration_proba_decay)

        if bool_debug:
            print(self.epsilon) 

    def my_loss_fn(self,y_true, y_pred):
         
        s1,s2=y_true.shape
        #print(s1,s2)
         
        # this matrix defines indices of a set of entries that we want to 
        # extract from y_true and y_pred
        # s2=2
        # s1=self.batchReplayBufferSize
        indices=np.zeros(shape=(s1,s2))
        indices[:,0]=np.arange(s1)
        indices[:,1]=self.actionsAppend
         
        # gather_nd and mean_squared_error are TensorFlow functions
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
        model.compile(optimizer = RMSprop(), loss = self.my_loss_fn, metrics = ['accuracy'])
        return model




    def trainingEpisodes():
        # gu process: en esta seccion del entrenamiento, ejecutaremos las acciones permitidas a los gus.
        #ejecutamos accion del drone utilizando la heuristica epsilon-greedy
        #ejecutamos accion drone en simulador y obtenemos reward de accion , asi como nuevo estado
        # almacenamos transition en buffer
        #si el replay buffer ya tiene su capacidad maxima llena, entonces procedemos con el entrenamiento de la network

        pass

    def selectAction(self,state):
        """
        esta funcion permite calcular la nueva accion a tomar por el agente, de acuerdo a la epsilon-greedy policy
        """
        if np.random.random() < self.epsilon: # si rand num menor a epsilon ejecutamos accion greedy
            return self.action_space[np.random.choice(np.arange(self.action_space.shape[0]),size = 1)[0],:]
        else:
            # calculamos el valor q(state,action) utilizando main network
            q_values =self.q_network.predict(state) 
            return self.action_space[np.argmax(q_values),:]

    def trainNetwork(self):
 
        # if the replay buffer has at least batch_size elements,
        # then train the model 
        # otherwise wait until the size of the elements exceeds batchReplayBufferSize
        if (len(self.replayBuffer)>self.batchReplayBufferSize):
             
 
            # sample a batch from the replay buffer
            randomSampleBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize)
             
            # here we form current state batch 
            # and next state batch
            # they are used as inputs for prediction
            currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,4))
            nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,4))            
            # this will enumerate the tuple entries of the randomSampleBatch
            # index will loop through the number of tuples
            for index,tupleS in enumerate(randomSampleBatch):
                # first entry of the tuple is the current state
                currentStateBatch[index,:]=tupleS[0]
                # fourth entry of the tuple is the next state
                nextStateBatch[index,:]=tupleS[3]
             
            # here, use the target network to predict Q-values 
            QnextStateTargetNetwork=self.targetNetwork.predict(nextStateBatch)
            # here, use the main network to predict Q-values 
            QcurrentStateMainNetwork=self.mainNetwork.predict(currentStateBatch)
             
            # now, we form batches for training
            # input for training
            inputNetwork=currentStateBatch
            # output for training
            outputNetwork=np.zeros(shape=(self.batchReplayBufferSize,2))
             
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
                 
                # this is necessary for defining the cost function
                self.actionsAppend.append(action)
                 
                # this actually does not matter since we do not use all the entries in the cost function
                outputNetwork[index]=QcurrentStateMainNetwork[index]
                # this is what matters
                outputNetwork[index,action]=y
             
            # here, we train the network
            self.mainNetwork.fit(inputNetwork,outputNetwork,batch_size = self.batchReplayBufferSize, verbose=0,epochs=100)     
             
            # after updateTargetNetworkPeriod training sessions, update the coefficients 
            # of the target network
            # increase the counter for training the target network
            self.counterUpdateTargetNetwork+=1 
            if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                # copy the weights to targetNetwork
                self.targetNetwork.set_weights(self.mainNetwork.get_weights())        
                print("Target network updated!")
                print("Counter value {}".format(self.counterUpdateTargetNetwork))
                # reset the counter
                self.counterUpdateTargetNetwork=0