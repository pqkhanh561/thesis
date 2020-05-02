#!/usr/bin/python3.7
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import subprocess
from env import env
from tqdm import tqdm
import time
import tensorflow as tf
from model import model

import socket   
import time     

# Import socket module
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Create a socket object
host = socket.gethostname() # Get local machine name
print(host)
port = 12345    # Reserve a port for your service.
s.bind((host, port))    # Bind to the port

s.listen(5) # Now wait for client connection.
c, addr = s.accept()    # Establish connection with client.
print ('Got connection from', addr)

env = env(c)
np.random.seed(0)

""" Implementation of deep q learning algorithm """ 
class DQN:

        def __init__(self, action_space, state_space):

                self.action_space = action_space
                self.state_space = state_space
                self.epsilon = 1
                self.gamma = .95
                self.batch_size = 64
                self.epsilon_min = .01
                self.epsilon_decay = .995
                self.learning_rate = 0.001
                self.memory = deque(maxlen=100000)
                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                self.model =model(state_space, action_space, self.sess)

        def build_model(self):
                return model

        def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))

        def act(self, state):

                if np.random.rand() <= self.epsilon:
                        return random.randrange(self.action_space)
                act_values = self.model.predict(state)
                return np.argmax(act_values[0])

        def replay(self):
                if len(self.memory) < self.batch_size:
                        return
                minibatch = random.sample(self.memory, self.batch_size)
                states = np.array([i[0] for i in minibatch])
                actions = np.array([i[1] for i in minibatch])
                rewards = np.array([i[2] for i in minibatch])
                next_states = np.array([i[3] for i in minibatch])
                dones = np.array([i[4] for i in minibatch])
                states = np.squeeze(states)
                next_states = np.squeeze(next_states)

                targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
                targets_full = self.model.predict_on_batch(states)
                #print(targets_full)
                ind = np.array([i for i in range(self.batch_size)])
                targets_full[[ind], [actions]] = targets
                #print('target', targets.shape)
                #print('target_full',targets_full.shape)
                #print('state',states.shape)

                ind = np.array([i for i in range(self.batch_size)])
                targets_full[[ind], [actions]] = targets
                self.model.fit(states, targets_full, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

def train_dqn(episode):
        action_dict = {0:'left', 1:'down', 2:'right', 3:'up', 4:'stay'}
        loss = []
        agent = DQN(4, 10)
        for e in range(episode):
                print("Episode {}".format(e))
                #state = env.reset() #################
                state = np.random.rand(1,10)
                state = np.reshape(state, (1, 10))
                score = 0
                max_steps = 1000
                full_msg=''
                for i in range(max_steps):
                        action = agent.act(state)
                        for i in range(11):
                                reward, next_state, done, full_msg = env.step(action,full_msg)
                                #time.sleep(2)
                       #print(next_state)
                        score += reward

                        #print([action_dict[action], done, reward])
                        next_state = np.reshape(next_state, (1, 10))
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        agent.replay()
                        if done:
                                print("")
                                print("episode: {}/{}, score: {}".format(e, episode, score))
                                time.sleep(2)
                                break
                loss.append(score)
        return loss


if __name__ == '__main__':
        ep = 10000
        loss = train_dqn(ep)
        plt.plot([i for i in range(ep)], loss)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.show()
