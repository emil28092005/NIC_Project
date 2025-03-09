import pygame
import sys
from gameobjects import *

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NIC_Project")

WHITE = (255, 255, 255)
RED = (255, 0, 0)


grass = Background("images/grass.jpg", WIDTH/2, HEIGHT/2)
horse1 = Horse("images/horse.png",50, 50)
barrier1 = Barrier("images/barrier.png", 400, 300)

horse1.set_size(75, 75)

grass.set_size(WIDTH, HEIGHT)

gameobjects = [grass, horse1, barrier1]
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    for object in gameobjects:
        object.draw(screen)
    pygame.display.flip()
    
    pygame.time.Clock().tick(60)
        