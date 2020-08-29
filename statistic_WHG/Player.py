from Dot import Dot
from Point import Point
import numpy as np
import pygame

np.random.seed(5)

action_arr = ['up', 'down', 'left', 'right', 'stay']

class Player:
    def __init__(self, x, y, tile):
        self.x = x
        self.y = y
        self.x_init = x;
        self.y_init = y;
        self.dead = False
        self.tile = tile
        self.action = None
        self.win = False

    def __str__(self):
        return('Agent postion: {}, {} '.format(self.x, self.y))

    def checkWin(self):
        if self.tile[self.x, self.y] == 3:
            self.win = True
            return True
        return False

    def move(self, action):
        if action=='up':
            self.y-=1
        elif action=='down':
            self.y+=1
        elif action=='left':
            self.x-=1
        elif action=='right':
            self.x+=1
        elif action=='stay':
            return


    def try_move(self):
        x = self.x
        y = self.y
        self.move(self.action)
        x_tmp = self.x
        y_tmp = self.y
        self.x = x
        self.y = y
        return x_tmp, y_tmp

    def reset(self, init_pos=False):
        if init_pos:
            self.x = np.random.choice(range(4,14))
            self.y = np.random.choice(range(5,8))
        else:
            self.x = self.x_init
            self.y = self.y_init
        self.action = None
        self.dead = False
        self.win = False

    def checkMove(self, action):
        x = self.x
        y = self.y
        self.move(action)
        if self.tile[self.x, self.y] == 0 :
            self.x = x
            self.y = y
            return False
        self.x = x
        self.y = y

        return True
   
    def checkCollapseDots(self, x, y, dots):
        for dot in dots:
            if x == dot.x and y == dot.y:
                return True
        return False

    def checkSpecial(self, dots):
        x, y  = self.try_move()
        for dot in dots:
            xd, yd = dot.try_update()
            if xd == self.x and yd == self.y:
                if x == dot.x and y == dot.y:
                    return True 
        return False 

    def move_step(self):
         if self.checkMove(self.action):
            self.move(self.action)

    def sample_move(self, dots):
        action = np.random.choice(action_arr)
        self.action = action 
#        self.action = 'right'
        
    def draw(self, screen):
        color = (255, 0, 0)
        pygame.draw.rect(screen, (0,0,0), (self.x*40, self.y*40, 35, 35))
        pygame.draw.rect(screen, color, (self.x*40, self.y*40, 32, 32))
    
    def getTile(self):
        return(self.tile[self.x, self.y])

#if __name__ == '__main__':
#    player = Player(4,7)
#    dot = Dot(13,7, Point(4,7), Point(13,7), 1, True, False)
#    for _ in range(100):
#        #action = np.random.choice(action_arr)
#        action = 'right' 
#        print(player)
#        print(dot)
#        player.move(action)
#        if player.collapse(dot):
#            print('DUNGGGGGGGGGGGGGGGGGGGGG')
