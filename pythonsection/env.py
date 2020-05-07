import subprocess
import time
import numpy as np

HEADERSIZE = 70
ACTION_DICT = ['up   ','down ','right','left ','stay ']

class env():
        def __init__(self, socket):
                self.state = None #Include position of agen and position of enemy
                self.reward = 0
                self.dead = 0   #1: is dead
                self.win = 0
                self.socket = socket
                
        def step(self, action, full_msg):
                #TODO: attach action to send to remote 
                #Require: action text need have length 5
                
                self.socket.send(bytes(ACTION_DICT[action],encoding='utf8'))
                while True:
                        t = self.socket.recv(128)
                        full_msg =full_msg + t.decode("utf-8")
                        if len(full_msg) > HEADERSIZE-1:
                                break
                msg = full_msg[:HEADERSIZE]
                full_msg = full_msg[HEADERSIZE:]
                #print(full_msg)
                #print(msg)
                #print(len(msg))
                

                #Test 
                '''
                self.state = np.random.rand(1,10)
                return self.reward, self.state, self.dead, full_msg
                '''

                #msg is contain all feature
                #2 is position of agent
                #8 next is position of enemy
                #1 next is agent dead
                #1 next is is win 
                list_feature = msg.split(',')
                self.state = [float(i) for i in list_feature[0:10]]
                self.dead = float(list_feature[10])
                self.win = float(list_feature[11])
                
                if self.dead==1:
                        self.reward-=3
                if self.win==1:
                        self.reward+=100
                        self.dead=1
                self.reward-=0.1

                return self.state, self.reward, self.dead, 'live' ,full_msg

        def preprocess_statefile(self, filestate):
                f = open(filestate, "r")
                pos_a = f.readline()
                pos_e = f.readline()
                pos_e = pos_e[:(len(pos_e)-2)]
                is_dead = f.readline()
                is_win = f.readline()
                f.close()
                pos_a = pos_a + ',' + pos_e
                self.state.append([float(i) for i in pos_a.split(",")])
                #self.state.append([float(i) for i in pos_e.split(",")])
                self.dead = float(is_dead)
                self.win = float(is_win)

        def reset(self):
                state,_,_,_,_ = self.step(4,'') #stay
                return(state)


#if __name__=="__main__":
#        e= env()
#        e.reset()
#        while(True):
#                e.step(1);
#                time.sleep(0)
