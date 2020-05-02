import subprocess
import time
import numpy as np

HEADERSIZE = 70

class env():
        filestate = "../state.txt"

        def __init__(self, socket):
                self.state = [] #Include position of agen and position of enemy
                self.reward = 0
                self.dead = 0   #1: is dead
                self.win = 0
                self.socket = socket
                #for i in range(50):
                #        self.socket.send(bytes('right',encoding='utf8'))
                #        tmp =''
                #        while True:
                #                t = self.socket.recv(16)
                #                tmp = tmp + t.decode("utf-8")
                #                if tmp.count(',')>=3 and tmp[-1]!=',':
                #                        break
                #try:
                #        while len(self.socket.recv(1024)):
                #            pass
                #except:
                #        print("empty")
                #        pass


        def step(self, action, full_msg):
                self.socket.send(bytes('right',encoding='utf8'))
                while True:
                        t = self.socket.recv(128)
                        full_msg =full_msg + t.decode("utf-8")
                        if len(full_msg) > HEADERSIZE-1:
                                break
                msg = full_msg[:HEADERSIZE]
                full_msg = full_msg[HEADERSIZE:]
                #print(full_msg)
                print(msg)
                #print(len(msg))
                print('')
                print('')
                self.state = np.random.rand(1,10)
                return self.reward, self.state, self.dead, full_msg
                '''
                
                #Change old state to new state
                oldstate = self.state
                while (True):
                        try:
                                self.reward = 0
                                self.dead= 0
                                self.state = []
                                self.win = 0

                                self.preprocess_statefile(self.filestate)
                                #print(self.state)
                                #print(self.dead)
                                #print("+++++++++++++++++++++++++")
                                break
                        except:
                                continue

                if self.dead==1:
                        self.reward-=3
                if self.win==1:
                        self.reward+=100
                        self.dead=1
                self.reward-=0.1
                #print(self.state[0][0:3])
                #print(oldstate)
                if oldstate != []:
                        if self.state[0][0:1] == oldstate[0][0:1] :
                                self.reward-=10
                subprocess.call('rm ' + self.filestate,shell=True)
                subprocess.call('touch ' + self.filestate, shell=True)
                '''
                return self.reward, self.state, self.dead



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
                _,state,_ = self.step(4) #stay
                return(state)


#if __name__=="__main__":
#        e= env()
#        e.reset()
#        while(True):
#                e.step(1);
#                time.sleep(0)
