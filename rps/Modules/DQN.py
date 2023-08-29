"""
este modulo contendra todas las clases y funciones que permitan desarrollar el entrenamiento por DQN de nuestro agente.
"""
from collections import namedtuple, deque
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop 
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error
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
    
class DQNAgent:
    def __init__(self,action_space, mem_capacity,gamma,epsilon,num_episodes, batch_size ) -> None:
        #env
        self.action_space = action_space
        self.memoryBuffer = ReplayMemory(mem_capacity)
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.batch_size = batch_size

        #suma de las recompensas por episodio
        self.sumRewardsEpisode = []

        #create q network
        self.q_network = self.createNetwork()

        #create target network
        self.target_network = self.createNetwork()
        self.action_space.set_weights(self.q_network.get_weights())




    def loss_fn():
        pass

    #create neural network
    def createNetwork(self):
        """
        esta funcion permitira crear la estructura de las redes DNN para este agente DQN
        """
        model=Sequential()
        model.add(Dense(128,input_dim=self.action_space.shape[0],activation='relu')) #añadir aqui state dimmension
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

    def selectAction():
        pass

    def trainNetwork():
        pass