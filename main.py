import pygame
import sys
from gameobjects import *
from genetic_alg import GeneticAlgorithm

POPULATION_SIZE = 50
MUTATION_RATE = 0.5
POPULATION_BEST = 0.2

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Initialize the screen
pygame.display.set_caption("NIC_Project")  # Set window title

WHITE = (255, 255, 255)  # Define white color

UI = UI()  # Initialize UI
grass = Background("images/Grass.jpg", 0, 0)  # Create background
horse1 = Horse("images/Horse_1.png", 50, 50)  # Create a horse
barrier1 = Barrier("images/Barrier.png", 400, 300)  # Create a barrier

BARRIER_SPEED = 1  # Speed of the barriers

grass.set_size(WIDTH, HEIGHT)  # Set background size

gameobjects = []  # List of all game objects
horses = [Horse("images/Horse_1.png", 50, HEIGHT/2 + 0 * i) for i in range(50)]  # Create 50 horses
barriers = [barrier1]  # List of barriers

gameobjects.extend([grass])  # Add background to game objects
gameobjects.extend(horses)  # Add horses to game objects
gameobjects.extend(barriers)  # Add barriers to game objects

upper_bound_rect = pygame.Rect(0, 0, WIDTH, -10), 
lower_bound_rect = pygame.Rect(0, HEIGHT, WIDTH, 10), 

spawner = Spawner("images/Barrier.png")  # Initialize spawner
last_barrier = barriers[0]

def get_features(horse):
    features = [last_barrier.rect.topleft[0], last_barrier.rect.topleft[1],
               last_barrier.rect.bottomright[0], last_barrier.rect.bottomright[1], 
               horse.rect.topleft[0], horse.rect.topleft[1],
               horse.rect.bottomright[0], horse.rect.bottomright[1],
               HEIGHT, BARRIER_SPEED, 0]
    return features 

genecticAlg = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, POPULATION_BEST, 10)

UI.add_horses(horses)  # Add horses to UI
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Handle window close event
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()  # Get pressed keys
    
    for i, horse in enumerate(horses):
        if horse.stopped == False:  # If the horse is not stopped
            data = get_features(horse)
            res = genecticAlg.predict(data, i)
            if res == 0:
                horse.up()
            elif res == 1:
                horse.down()
            else:
                horse.stay()
        else:
            horse.move(-BARRIER_SPEED, 0)  # Move stopped horse to the left

        horse.apply_vacceleration()  # Apply acceleration to horse
        horse.apply_vspeed()  # Apply speed to horse
        horse.draw(screen)  # Draw the horse

    spawner.handle()  # Handle spawner logic
    if spawner.tick_counter == 0:  # If a new barrier is spawned
        new_barrier = spawner.spawn()  # Spawn a new barrier
        barriers.append(new_barrier)  # Add barrier to barriers list
        gameobjects.append(new_barrier)  # Add barrier to game objects
        last_barrier = new_barrier

    for barrier in barriers:
        barrier.move(-BARRIER_SPEED, 0)  # Move barriers to the left

    for horse in horses:
        horse.update_animation()
        if horse.stopped == False:  # Check for collisions between horses and barriers
            for barrier in barriers:
                if horse.rect.colliderect(barrier.rect):  # If collision detected
                    horse.stop()  # Stop the horse
            if horse.rect.colliderect(upper_bound_rect) or horse.rect.colliderect(lower_bound_rect):
                horse.stop()
            horse.count_fitness
                

    for object in gameobjects:
        object.draw(screen)  # Draw all game objects
    UI.draw_marks(screen)  # Draw UI marks

    pygame.display.flip()  # Update the display
    pygame.time.Clock().tick(160)  # Limit the frame rate to 160 FPS