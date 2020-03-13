import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import subprocess
from env import env
from tqdm import tqdm
import time
import tensorflow as tf

env = env()
np.random.seed(0)

class DQN:

	""" Implementation of deep q learning algorithm """

	def __init__(self, action_space, state_space):

		self.action_space = action_space
		self.state_space = state_space
		self.epsilon = 1
		self.gamma = .95
		self.batch_size = 10
		self.epsilon_min = .01
		self.epsilon_decay = .995
		self.learning_rate = 0.001
		self.memory = deque(maxlen=100000)
		self.model = self.build_model()

	#Give session 
	def set_session(self,session):
		self.sess = session

	def build_model(self):
        with tf.device('/cpu:0'):
            X = tf.placeholder(tf.float32, [None, num_input])
            Y = tf.placeholder(tf.float32, [None, num_classes])

		model = tf.keras.models.Sequential()
		model.add(tf.keras.layers.Dense(64, input_shape=(self.state_space,), activation='relu'))
		model.add(tf.keras.layers.Dense(64, activation='relu'))
		model.add(tf.keras.layers.Dense(self.action_space, activation='linear'))
		model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
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

		ind = np.array([i for i in range(self.batch_size)])
		targets_full[[ind], [actions]] = targets

		self.model.fit(states, targets_full, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

def train_dqn(episode):
	loss = []
	agent = DQN(4, 10)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		agent.set_session(sess)
		for e in range(episode):
			print("Episode {}".format(e))
			state = env.reset()
			state = np.reshape(state, (1, 10))
			score = 0
			max_steps = 1000
			for i in tqdm(range(max_steps)):
				action = agent.act(state)
				for i in range(11):
					reward, next_state, done = env.step(action)
				score += reward
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
