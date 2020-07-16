import gym
import sys 
import random 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime
from tqdm import tqdm
from env import env
import pandas as pd

MAX_EXPERIENCES = 1000000 #500000
MIN_EXPERIENCES = 50
TARGET_UPDATE_PERIOD = 10000
K = 5
MAX_STEP=100
NUM_FRAME = 5e6

np.random.seed(1)

import time     
global env 

def learn(model, targets_model, experience_replay_buffer, gamma, batch_size):
    samples = random.sample(experience_replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
    next_Qs = targets_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

    loss = model.update(states, actions, targets)
    return loss

class DQN:
    def __init__(self, K, scope, input_shape, save_path = 'model/env.ckpt'):
        self.K = K
        self.scope = scope
        self.save_path = save_path

        with tf.variable_scope(scope):

            #inputs and targets
            self.X = tf.placeholder(tf.float32, shape=(None, input_shape), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')


            fc1 = tf.contrib.layers.fully_connected(self.X, 128, activation_fn=tf.nn.relu)
            #fc2 = tf.contrib.layers.fully_connected(fc1, 32, activation_fn=tf.nn.relu)

            self.predict_op = tf.contrib.layers.fully_connected(fc1, K)

            selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, K), reduction_indices=[1])

            self.cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
            self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

    def update(self, states, actions, targets):
        c, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.X: states,
                self.G: targets,
                self.actions: actions
                }
            )
        return c

    def set_session(self, sess):
        self.session = sess

    def predict(self, states):
        tmp = self.session.run(self.predict_op, feed_dict={self.X:states})
        #return self.session.run(self.predict_op, feed_dict={self.X:states})
        return tmp
    
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])
            print(x) 
            print(self.session.run(self.fc1, feed_dict={self.X:[x]}))
            print(self.session.run(self.fc2, feed_dict={self.X:[x]}))
            print(self.session.run(self.predict_op, feed_dict={self.X:[x]}))
            print("")
            print("")

    def load(self):
        self.saver = tf.train.Saver(tf.global_variables())
        load_was_success = True
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.session, load_path)
        except:
            print("No saved model to load. Start new one")
            load_was_success = False
        else:
            print("Load model: {}".format(load_path))
            saver = tf.train.Saver(tf.global_variables())
            episode_number = int(load_path.split('-')[-1])  #what?

    def save(self, n):
        self.saver.save(self.session, self.save_path, global_step=n)
        print("SAVE MODEL #{}".format(n))

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v:v.name)
        others = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        others = sorted(others, key=lambda v: v.name)

        ops=[]
        for p,q in zip(mine, others):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        
        self.session.run(ops)


def trainning():
    num_action_act = [0,0,0,0,0]
    gamma = 0.99
    batch_sz = 32 
    num_episodes = 1000000
    total_t = 0
    experience_replay_buffer = []
    episode_rewards = []
    last_100_avgs = []
    max_eps = [-sys.maxsize]



    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 1000000#500000

    model = DQN(K=K, input_shape=4 + 3*number_enemy, scope="model")
    target_model = DQN(K=K, input_shape=4 + 3*number_enemy, scope="target_model")

    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())
        model.load()

        print("Filling experience replay buffer...")
        sate = env.reset()

        #for i in range(MIN_EXPERIENCES):
        for i in tqdm(range(MIN_EXPERIENCES)):
            action = np.random.randint(0,K)
            num_action_act[action] +=1
            state, reward, done, _ = env.step(action)

            done = done == 1
            #time.sleep(0.5)
            #print(obs)

            next_state = state
            experience_replay_buffer.append((state, action, reward, next_state, done))

            if done:
                state = env.reset()
            else:
                state = next_state

        print(num_action_act)
        #try:
        i = -1
        #for i in range(num_episodes):
        while True:
            i+=1
            t0 = datetime.now()
            
            state = env.reset()

            loss = None


            total_time_training = 0
            num_steps_in_episode = 0
            episode_reward = 0

            done = False
            while True:
                if total_t % TARGET_UPDATE_PERIOD == 0:
                    target_model.copy_from(model)
                    print("Copied model parameters to target network, total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))
                
                action = model.sample_action(state, epsilon)
                num_action_act[action] +=1
                time_act = datetime.now()
                next_state, reward, done, _  = env.step(action)
                time_act = datetime.now() - time_act

                done = done==1

                episode_reward += reward

                if len(experience_replay_buffer)==MAX_EXPERIENCES:
                    experience_replay_buffer.pop(0)
                experience_replay_buffer.append((state, action, reward, next_state, done))

                t0_2 = datetime.now()
                loss = learn(model, target_model, experience_replay_buffer, gamma, batch_sz)
                dt = datetime.now() - t0_2

            
                total_time_training += dt.total_seconds()
                num_steps_in_episode +=1
                
                state = next_state
                total_t += 1

                epsilon = max(epsilon - epsilon_change, epsilon_min)
                if done:
                    break

            duration = datetime.now() - t0

            episode_rewards.append(episode_reward)      #Reward every eps

            max_eps.append(max(max_eps[-1], episode_reward)) #Max eps

            time_per_step = total_time_training/num_steps_in_episode

            last_100_avg = np.array(episode_rewards[max(0, i-100):i]).mean()
            last_100_avgs.append(last_100_avg)          #Avg reward every eps

            
            print("Episode: {:>6}, Num steps: {:>3}, Reward: {:>5}, Avg reward: {:>5.3f}, Max: {:>5.3f} Eps: {:>5.3f}".format(i, num_steps_in_episode, episode_reward, last_100_avg, max_eps[-1], epsilon))
            if i % 50 ==0:
                model.save(i)
            sys.stdout.flush()
            if np.sum(num_action_act) > NUM_FRAME:
                break
        #except:
        #    print("Break")
        #finally:
        #    max_eps.pop(0)
        #    data = pd.DataFrame({'Reward': episode_rewards, 'Avg Reward': last_100_avgs, 'Max': max_eps})
        #    data.to_csv("./data_result.csv")

        #    figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

        #    plt.plot('Reward', '--', color="#999999", data = data, label="Reward")
        #    plt.plot('Avg Reward', data = data, label="Avg Reward")
        #    plt.plot('Max', data = data, label="Max")
        #    plt.legend(loc="upper left")

        #    plt.xlabel('episodes')
        #    #plt.show()
        #    plt.savefig('result.png')
        #    print(num_action_act)
        #    #env.close()

def testing(): 
    gamma = 0.99
    batch_sz = 32 
    num_episodes = 1000 
    total_t = 0
    episode_rewards = np.zeros(num_episodes)
    last_100_avgs = []

    model = DQN(K=K, input_shape=2 + 2*number_enemy, scope="model")

    with tf.Session() as sess:
        model.set_session(sess)
        sess.run(tf.global_variables_initializer())
        model.load()

        state = env.reset()
        
        acc = [1,2,2,2,2,2,2,0,2,2,2,2,2,2,2] 
        for i in range(int(10e6)):
            if i % 4 == 0:
                action = model.sample_action(state, 0.1)
            else:
                action = model.sample_action(state, 0.1)
            #action = acc[i%len(acc)]
            next_state, reward, done, _ = env.step(action)
            #time.sleep(0.8)
            #print(action)
            done = done == 1

            if done:
                obs = env.reset()
                print("Reward: {}".format(i,reward))
                state = obs
            else:
                state = next_state




if __name__ == "__main__":
    number_enemy = int(sys.argv[1])
    env = env(number_enemy)
    if sys.argv[2] == "train":
        print("train")
        trainning()
    elif sys.argv[2]=="test":
        print("test")
        testing()
