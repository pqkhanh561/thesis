import time
import numpy as np
import math
from Game import WorldHardestGame 
import pandas as pd

ACTION_DICT = ['up','down','right','left','stay','reset']
game = WorldHardestGame()

class env():
        def __init__(self, number_enemy, action_space=4, seed=1):
                self.state = None#Include position of agen and position of enemy
                self.reward = 0
                self.dead = 0   #1: is dead
                self.win = 0
                self.num_enemy = number_enemy
                self.action_space=5
                self.agent_pos = []

        def seed(self, seed):
                np.random.seed(seed)
                
        def step(self, action):
#                time.sleep(0.4)
                self.reward = 0
                self.dead = 0
                self.win = 0
                self.state = None
                #Require: action text need have length 5
                
                list_feature = game.run(ACTION_DICT[action], False)
               
                #msg is contain all feature
                #2 is position of agent
                #8 next is position of enemy
                #1 next is agent dead
                #1 next is is win # type of tile that agent is standing 
                self.state = [float(i) for i in list_feature[0:(3*self.num_enemy+1)+1]]

                #Simple state
                #for i in range(2,10):
                #    self.state[i] = int(self.state[i])

                #for i in range(2,10):
                #    if i%2 == 0:
                #        self.state[i] = self.state[0] - self.state[i]
                #    else:
                #        self.state[i] = self.state[1] - self.state[i]

                self.agent_pos.append((self.state[0], self.state[1]))
                pd.DataFrame({'Agent' : self.agent}).to_csv('./agent_pos.csv')

                self.dead = float(list_feature[14])
                self.win = float(list_feature[15])
                tile = float(list_feature[16])
               
                #REWARD for env
                #self.state.append(15-self.state[0])
                #if self.win!=1:
                #        self.state.append(self.dead)
                #else:
                #        self.state.append(2)

                if self.dead==1 and self.win!=1:
                        self.reward-=50
                if self.win==1:
                        self.reward+=100
                        self.dead=1
                if tile==1 or tile==2:
                        self.reward-=0
                        self.reward-=1/(1 + abs(11-15+self.state[0]) + abs(4 - 7+ self.state[1]))

                #elif tile==2:
                #        self.reward-=0.1
                
                #print('State: {}, reward: {}, dead: {}'.format(self.state, self.reward, self.dead))
                return self.state, self.reward, self.dead, {}

        def reset(self):
                state,_,_,_ = self.step(5)#stay
                return(state)


if __name__=="__main__":
        e= env(4)
        e.reset()
        while(True):
                e.step(np.random.randint(4))
                print(e.state)
