
from Point import Point 
import pygame

class Dot:
    def __init__(self, x, y, pos1, pos2, speed,  moveToPos1, vertMovement):
        self.x = x
        self.y = y
        self.x_init = x 
        self.y_init = y
        self.pos1 = pos1
        self.pos2 = pos2
        self.speed = speed
        self.moveToPos1 = moveToPos1
        self.vertMovement = vertMovement

    def __str__(self):
        return("Dot position: {} , {}".format(self.x, self.y))

    def reset(self):
        self.x = self.x_init
        self.y = self.y_init

    def update(self):
        if self.moveToPos1:
            if not self.vertMovement:
                self.x -= self.speed
            else:
                self.y -= self.speed
            if self.x < self.pos1.x  or self.y < self.pos1.y:
                self.moveToPos1= False 
        else:
            if not self.vertMovement:
                self.x += self.speed
            else:
                self.y += self.speed
            if self.x > self.pos2.x or self.y > self.pos2.y:
                self.moveToPos1= True 

    def try_update(self):
        x = self.x
        y = self.y

        if self.moveToPos1:
            if not self.vertMovement:
                x -= self.speed
            else:
                y -= self.speed
        else:
            if not self.vertMovement:
                x += self.speed
            else:
                y += self.speed
        return x, y

    def getX(self):
        return self.x

    def getY(self):
        return self.y
    

    def draw(self, screen):
        pygame.draw.circle(screen, (0,0,0), (self.x*40+20, self.y*40+20), 16)
        pygame.draw.circle(screen, (0,0,255), (self.x*40+20, self.y*40+20), 14)

#if __name__=='__main__':
#    dot = Dot(13,5, Point(6,5), Point(13,5), 1, True, False)
#    for _ in range(100):
#        dot.update()
#        print(dot)
