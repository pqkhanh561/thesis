import time
import numpy as np
import math
from Game import WorldHardestGame 

ACTION_DICT = ['up','down','right','left','stay','reset']
np.random.seed(1)
game = WorldHardestGame()

class env():
        def __init__(self, number_enemy):
                self.state = None #Include position of agen and position of enemy
                self.reward = 0
                self.dead = 0   #1: is dead
                self.win = 0
                self.num_enemy = number_enemy
                
        def step(self, action):
                self.reward = 0
                self.dead = 0
                self.win = 0
                self.state = None
                #Require: action text need have length 5
                
                full_msg = game.run(ACTION_DICT[action])
               
                #msg is contain all feature
                #2 is position of agent
                #8 next is position of enemy
                #1 next is agent dead
                #1 next is is win 
                # type of tile that agent is standing
                list_feature = full_msg
                self.state = [float(i) for i in list_feature[0:(2*self.num_enemy+1)+1]]

                #Simple state
                #for i in range(2,10):
                #    self.state[i] = int(self.state[i])
               

                self.dead = float(list_feature[10])
                self.win = float(list_feature[11])
                tile = float(list_feature[12])
               
                #REWARD for env
                if self.dead==1 and self.win!=1:
                        self.reward-=50
                if self.win==1:
                        self.reward+=200
                        self.dead=1
                if tile==1:
                        self.reward-=0
                        self.reward+=(11-(15-self.state[0]))

                elif tile==2:
                        self.reward-=0
                
                #print('State: {}, reward: {}, dead: {}'.format(self.state, self.reward, self.dead))
                return self.state, self.reward, self.dead, 'live' 

        def reset(self):
                state,_,_,_ = self.step(5)#stay
                return(state)


if __name__=="__main__":
        e= env(4)
        e.reset()
        while(True):
                e.step(np.random.randint(4));
