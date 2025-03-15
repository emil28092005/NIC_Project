import pygame
import sys
from gameobjects import *

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NIC_Project")

WHITE = (255, 255, 255)

UI = UI()
grass = Background("images/grass.jpg", 0, 0)
horse1 = Horse("images/horse.png", 50, 50)
barrier1 = Barrier("images/barrier.png", 400, 300)

BARRIER_SPEED = 1

grass.set_size(WIDTH, HEIGHT)

gameobjects = [grass, horse1, barrier1]
horses = [Horse("images/horse.png", 50, 50) for i in range(5)]
barriers = [barrier1]

UI.add_children(horses)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    
    for horse in horses:
        if horse.stopped == False:
            if keys[pygame.K_UP]:    
                horse.up()           
            elif keys[pygame.K_DOWN]:
                horse.down()         
            else:
                horse.stay()    
        else:
            horse.move(-BARRIER_SPEED, 0)    

        horse.apply_vacceleration()
        horse.apply_vspeed()         
        horse.draw(screen)           

    for barrier in barriers:
        barrier.move(-BARRIER_SPEED, 0)

    # Checking collisions between horses and barriers
    for horse in horses:
        if horse.stopped == False:
            for barrier in barriers:
                if horse.rect.colliderect(barrier.rect):
                    horse.stop()
                

    for object in gameobjects:
        object.draw(screen)
    UI.draw_marks(screen)

    pygame.display.flip()
    pygame.time.Clock().tick(60)
