import numpy as np
from Dot import Dot
from Point import Point
from Player import Player
import pygame
import time
from tqdm import tqdm

np.random.seed(5)

action_arr = ['up', 'down', 'left', 'right', 'stay']
class WorldHardestGame:

    def __init__(self, render= False):
        self.map = None
        self.player = None
        self.dots = []
        self.readResource() 
        self.can_move=True
        self.render = render
        self.num_win = 0 
        try:
                pygame.init()
                if self.render:
                        self.screen = pygame.display.set_mode((800, 600))
        except:
                pass


    def readResource(self):
        f = open("./map/level_1.txt", "r")

        #READ MAP
        arr = []
        for _ in range(15):
            tmp = f.readline()
            tmp = tmp.replace('\n','')
            arr.append(list(map(int,list(tmp))))

        arr = np.array(arr)
        arr = arr.reshape(15,20)
        self.map = arr.T
        #print(arr)



        f.readline()
        f.readline()
        f.readline()
        f.readline()


        #READ DOTS
        while True:
            tmp = f.readline()
            tmp = tmp.replace('\n','')
            if tmp == '':
                break
            tmp = tmp.split(' - ')
            point1 = list(map(int,tmp[2].split(',')))
            point2 = list(map(int,tmp[3].split(',')))
            self.dots.append( Dot(  int(tmp[0]), int(tmp[1]), Point(point1[0], point1[1]), Point(point2[0], point2[1]), int(tmp[4]),
                                    tmp[5]=='true',
                                    tmp[6]=='true'))
        f.close()
        
        #Print Dots
        #for dot in self.dots:
        #    print(dot)

        #READ PLAYER
        f = open("./map/level_1.properties", "r")
        f.readline()
        tmp = list(f.readline().split('='))
        tmp = tmp[1].replace('\n','').split(',')
        self.player = Player(int(tmp[0]), int(tmp[1]),arr.T)

    def reset(self, init_pos):
        self.player.dead = False
        if not init_pos:
            self.player.reset()
            for dot in self.dots:
                dot.reset()
        else:
            self.player.reset(init_pos)
            while True:
               break_init = True 
               self.dots[0].reset(init_pos)
               self.dots[2].copy(self.dots[0])
               self.dots[1].x = 9 - (self.dots[0].x - 10)
               self.dots[1].moveToPos1 = not self.dots[0].moveToPos1
               self.dots[3].copy(self.dots[1])
               for dot in self.dots:
                    if dot.x == self.player.x and dot.y == self.player.y:
                        break_init=False
                        break 
               if break_init:
                    break 
        
    def drawTile(self):
        self.screen.fill((0,0,0))
        for x in range(20):
            for y in range(15):
                if self.map[x,y]==1:
                    if x % 2 == 0:
                        if y % 2 == 0:
                            color = (230, 230, 255)
                        else:
                            color = (255,255,255)
                    elif x % 2 == 1:
                        if y % 2 != 0:
                            color = (230, 230, 255)
                        else:
                            color = (255,255,255)
                    pygame.draw.rect(self.screen, color, (x*40, y*40, 40, 40))
                elif self.map[x,y]==0:
                    color = (204,204,255)
                    pygame.draw.rect(self.screen, color, (x*40, y*40, 40, 40))
                elif self.map[x,y]==2:
                    color = (181, 254, 80) 
                    pygame.draw.rect(self.screen, color, (x*40, y*40, 40, 40))
                elif self.map[x,y]==3:  #GOAL
                    color = (181, 254, 180)
                    pygame.draw.rect(self.screen, color, (x*40, y*40, 40, 40))

       
    def update(self):
        self.can_move = True
        #TODO: Action for agent
        if self.player.checkSpecial(self.dots):
            self.player.dead = True

        if self.render:
            self.drawTile()

        for dot in self.dots:
            if self.player.action!='reset':
                dot.update()
            if self.render:
                dot.draw(self.screen)

        if self.player.action!='reset':
            self.player.move_step()

        if self.render:
            self.player.draw(self.screen)
        
        #print(self.player)
        #for i in range(4):
        #    print('Dot {}: {}'.format(i,self.dots[i]))
        #print("")
        if self.player.checkCollapseDots(self.player.x, self.player.y, self.dots):
            self.player.dead = True
            self.can_move=False 

        if self.player.checkWin():
            self.num_win+=1
            self.player.dead = True
            #print(self.num_win)

        if self.render: pygame.display.flip() 
    #def draw(self):
    #    pass

    def getGameState(self):
        state = [self.player.x, self.player.y]
        for dot in self.dots:
            state.append(dot.x)
            state.append(dot.y)
            state.append(dot.moveToPos1)
        state.append(int(self.player.dead))
        state.append(int(self.player.win))
        state.append(self.player.getTile())
        return(state)

    def run(self, action, init_pos=True):
        if action=='reset':
            self.reset(init_pos)
            self.player.action = 'reset'
            self.update()
            return(self.getGameState())
        else:
            #self.player.sample_move(self.dots)
            self.player.action = action 
            #if self.player.win:
            #    time.sleep(1)
            #    print("WIN")
            res = self.getGameState()
            if self.player.dead:
                self.reset(init_pos)
            self.update()
            return(res)




if __name__ == '__main__':
    game = WorldHardestGame()
    acc = [1,3,3,3,3,3,3,0,3,3,3,3,3,3,3]
    for i in range(5000000):
        action = np.random.choice(4)
        #time.sleep(4)
       #print(game.run(action_arr[acc[i%len(acc)]])) 
        if game.player.win:
            time.sleep(1)
            print("WIN")
        print(game.run('reset'))
