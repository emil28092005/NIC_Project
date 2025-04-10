import pygame
import sys
from gameobjects import *
from genetic_alg import GeneticAlgorithm

POPULATION_SIZE = 20
MUTATION_RATE = 0.5
POPULATION_BEST = 0.2
FRAME_RATE = 160 #TODO: FIX THE INCORRECT FRAME RATE CORRELATION
BARRIER_SPEED = 10
BARRIER_DELAY = 75

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))  # Initialize the screen
pygame.display.set_caption("NIC_Project")  # Set window title

WHITE = (255, 255, 255)  # Define white color

UI = UI("images/HUD.png")  # Initialize UI
gameobjects = []
horses = []
barriers = []

upper_bound_rect = None
lower_bound_rect = None

spawner = None
last_barrier = None

def init_game():
    global gameobjects, horses, barriers, upper_bound_rect, lower_bound_rect, spawner, last_barrier, BARRIER_SPEED

    grass = Background("images/Grass.jpg", 0, 0)
    grass.set_size(WIDTH, HEIGHT)

    gameobjects = []
    
    barriers = []

    gameobjects = [grass]
    for horse in horses:
        horse.set_position(50, HEIGHT/2)
        horse.stopped = False
        horse.set_vacceleration(0)
        horse.set_vspeed(0)
        horse.frame_counter = 0
        horse.fitness = 0
    spawner = Spawner("images/Barrier.png", BARRIER_DELAY)
    new_barrier = spawner.spawn()
    barriers.append(new_barrier)
    gameobjects.extend(horses)
    gameobjects.extend(barriers)
    
    upper_bound_rect = pygame.Rect(0, 0, WIDTH, -10)
    lower_bound_rect = pygame.Rect(0, HEIGHT, WIDTH, 10)

    
    last_barrier = barriers[0]

    UI.add_horses(horses)

def get_features(horse):
    features = [last_barrier.rect.topleft[0], last_barrier.rect.topleft[1],
               last_barrier.rect.bottomright[0], last_barrier.rect.bottomright[1], 
               horse.rect.topleft[0], horse.rect.topleft[1],
               horse.rect.bottomright[0], horse.rect.bottomright[1],
               0, HEIGHT, BARRIER_SPEED]
    return features 

horses = [Horse("images/Horse_1.png", 50, HEIGHT/2 + 0 * i) for i in range(POPULATION_SIZE)]
init_game()
genecticAlg = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, POPULATION_BEST, len(get_features(horses[0])))
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
            horse.count_fitness()
                

    for object in gameobjects:
        object.draw(screen)  # Draw all game objects

    UI.draw(screen)
    UI.draw_marks(screen)  # Draw UI marks
    
    if all(horse.stopped for horse in horses):
        UI.iteration_num += 1
        genecticAlg.learn([x.fitness for x in horses])
        init_game()

    pygame.display.flip()  # Update the display
    pygame.time.Clock().tick(FRAME_RATE)  # Limit the frame rate to 160 FPS