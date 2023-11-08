"""
This module leverages utility functions for the training/testing of tf-agents in custom environments
"""

import base64
import imageio
import IPython
import io
import tensorflow as tf
import shutil
import zipfile
from tf_agents.trajectories import trajectory

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential,load_model,model_from_json
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error
import keras
import numpy as np

#create neural network
def createNetwork(state_dimension,hid_lay_neurons,action_dimmension,output_layer_activation_function = None, loss_input_function = None,compiler_metrics = None):
        """
        esta funcion permitira crear la estructura de las redes DNN para este agente DQN
        """
        hidden_layers_num = len(hid_lay_neurons)
        model=Sequential()
        model.add(Dense(hid_lay_neurons[0],input_dim = state_dimension,activation='relu'))         
        for neurons_num in range(1,hidden_layers_num):
          model.add(Dense(hid_lay_neurons[neurons_num],activation='relu'))
        model.add(Dense(action_dimmension,activation=output_layer_activation_function))
        # compile the network with the custom loss defined in my_loss_fn
        if loss_input_function:
          model.compile(optimizer = Adam(), loss = loss_input_function, metrics = compiler_metrics)
        return model

def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)


def create_policy_eval_video(policy, filename, eval_env,eval_py_env,num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for i in range(num_episodes):
        print(f"episode : {i}")
        time_step = eval_env.reset()
        images = eval_py_env.render()
        for im in images:
            video.append_data(im)
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            images = eval_py_env.render()
            for im in images:
                video.append_data(im)
        print("new episode...")
  return embed_mp4(filename)


def embed_gif(gif_buffer):
  """Embeds a gif file in the notebook."""
  tag = '<img src="data:image/gif;base64,{0}"/>'.format(base64.b64encode(gif_buffer).decode())
  return IPython.display.HTML(tag)

def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env):
  num_episodes = 3
  frames = []
  for i in range(num_episodes):
    print(f"episode : {i}")
    time_step = eval_tf_env.reset()
    frames.append(eval_py_env.render())
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = eval_tf_env.step(action_step.action)
      frames.append(eval_py_env.render())
    print("new episode...")
  gif_file = io.BytesIO()
  imageio.mimsave(gif_file, frames, format='gif', fps=60)
  IPython.display.display(embed_gif(gif_file.getvalue()))

def create_zip_file(dirname, base_filename):
  return shutil.make_archive(base_filename, 'zip', dirname)

def unzip_file_to(file_name,dirname):
  #if files is None:
  #  return
  #uploaded = files.upload()
  #for fn in uploaded.keys():
    #print('User uploaded file "{name}" with length {length} bytes'.format(
    #    name=fn, length=len(uploaded[fn])))
    shutil.rmtree(dirname)
    zip_files = zipfile.ZipFile(file_name,"r")#io.BytesIO(uploaded[fn]), 'r')
    zip_files.extractall(dirname)
    zip_files.close()

#-------------------------------------------------------- REWARD FUNCTIONS -------------------------------------- ################################################3

def rewardFunc(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_dr = conec_1 * (env.obj_gus.transmission_rate[0]/trans_rate_tot) + conec_2 *(env.obj_gus.transmission_rate[1]/
                            trans_rate_tot)

        sum_dis = ((env.obj_drones.rc - dis_1) / env.obj_drones.rc) + ((env.obj_drones.rc - dis_2) / env.obj_drones.rc)

        reward = weight_dr * sum_dr + weight_dis * sum_dis
        return reward



def rewardFunc2(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_conec = conec_1 + conec_2


        sum_dr = (env.obj_gus.transmission_rate[0]/trans_rate_tot) + (env.obj_gus.transmission_rate[1]/trans_rate_tot)

        sum_dis = ((env.obj_drones.rc - dis_1) / env.obj_drones.rc) + ((env.obj_drones.rc - dis_2) / env.obj_drones.rc)
        reward = sum_conec + weight_dr * sum_dr + weight_dis * sum_dis

        return reward


def rewardFunc3(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_dr = conec_1 * (env.obj_gus.transmission_rate[0]/trans_rate_tot) + conec_2 *(env.obj_gus.transmission_rate[1]/
                            trans_rate_tot)

        sum_dis = ((env.obj_drones.rc - dis_1) / env.obj_drones.rc) + ((env.obj_drones.rc - dis_2) / env.obj_drones.rc)

        reward = weight_dr * sum_dr #+ weight_dis * sum_dis
        return reward

def rewardFunc4(env,weight_dr,weight_dis):
        conec_1 = float(env.obj_drones.dict_gu[0]["Gu_0"]["Connection"])
        conec_2 = float(env.obj_drones.dict_gu[0]["Gu_1"]["Connection"])
        dis_1 = env.obj_drones.dict_gu[0]["Gu_0"]["DistanceToDrone"]
        dis_2 = env.obj_drones.dict_gu[0]["Gu_1"]["DistanceToDrone"]
        trans_rate_tot = env.obj_gus.transmission_rate[0] + env.obj_gus.transmission_rate[1] + 1e-6

        sum_dr = conec_1 * (env.obj_gus.transmission_rate[0]/trans_rate_tot) + conec_2 *(env.obj_gus.transmission_rate[1]/
                            trans_rate_tot)

        sum_dis = env.obj_drones.rc / (dis_1 + dis_2 + 0.001)

        reward = weight_dr * sum_dr + weight_dis * sum_dis
        return reward

#-------------------------------------------------------- REWARD FUNCTIONS -------------------------------------- ################################################3


def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

# functions to collect data to the replay buffer
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

def save_tf_weights_to_keras_model_weights(tf_kernel_biases, keras_model,filepath):
  #update keras model weights with trained tf model q network from tf-agents
  for i in range(len(keras_model.layers)):
      keras_model.layers[i].kernel.assign(tf_kernel_biases[2 *i])
      keras_model.layers[i].bias.assign(tf_kernel_biases[2 * i+1])

  #save keras model weights...
  keras_model.save_weights(filepath)


