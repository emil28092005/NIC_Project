import pygame
import sys
from gameobjects import *
from genetic_alg import GeneticAlgorithm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

POPULATION_SIZE = 50  # Number of horses in population
MUTATION_RATE = 0.5
POPULATION_NEW = 0.1
POPULATION_BEST = 0.3
FRAME_RATE = 300
BARRIER_SPEED = 10
BARRIER_DELAY = 100
MAX_ITERATIONS = 50
BARRIER_SEED = 42

current_iteration = 1
best_fitnesses = []
running_fitnesses = []

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))
pygame.display.set_caption("NIC_Project")
font = pygame.font.SysFont("Arial", 36)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TEXT_COLOUR = (10, 20, 200)

UI = UI("images/HUD.png")
gameobjects = []
horses = []
barriers = []
upper_bound_rect = None
lower_bound_rect = None
spawner = None
last_barrier = None


def init_game():
    global gameobjects, horses, barriers, upper_bound_rect, lower_bound_rect, spawner, last_barrier, BARRIER_SPEED
    # Initialize background and game objects for a new iteration
    #random.seed(BARRIER_SEED) #SET SEED
    grass = Background("images/Grass.png", 0, 0)
    grass.set_size(WIDTH, HEIGHT)

    gameobjects = [grass]
    barriers = []

    # Reset horses to starting position and state
    for horse in horses:
        horse.set_position(50, HEIGHT / 2)
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

    # Define upper and lower bounds for horse movement
    upper_bound_rect = pygame.Rect(0, 0, WIDTH, -10)
    lower_bound_rect = pygame.Rect(0, HEIGHT, WIDTH, 10)

    last_barrier = barriers[0]

    UI.add_horses(horses)


def get_features(horse: Horse):
    # Extract normalized features for neural network input
    features = [last_barrier.rect.topleft[0], last_barrier.rect.topleft[1],
                last_barrier.rect.bottomright[0], last_barrier.rect.bottomright[1],
                horse.rect.topleft[0], horse.rect.topleft[1],
                horse.rect.bottomright[0], horse.rect.bottomright[1],
                0, HEIGHT, horse.vspeed, horse.vacceleration, BARRIER_SPEED]
    features = [x / HEIGHT for x in features]
    return features


horses = [Horse("images/Horse_1.png", 50, HEIGHT / 2 + 0 * i) for i in range(POPULATION_SIZE)]

init_game()

genecticAlg = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE, POPULATION_BEST, POPULATION_NEW,
                                len(get_features(horses[0])))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save fitness plot on exit
            plt.plot(running_fitnesses)
            plt.xlabel("Iteration")
            plt.ylabel("Current Fitness")
            plt.title("Current Fitness per Iteration")
            plt.savefig('fitness_plot.png')
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()

    for i, horse in enumerate(horses):
        if not horse.stopped:
            data = get_features(horse)
            res = genecticAlg.predict(data, i)
            # Control horse movement based on neural network output
            if res == 0:
                horse.up()
            elif res == 1:
                horse.down()
            else:
                horse.stay()
        else:
            # Move stopped horses left with barriers
            horse.move(-BARRIER_SPEED, 0)

        horse.apply_vacceleration()
        horse.apply_vspeed()
        horse.draw(screen)

    spawner.handle()
    if spawner.tick_counter == 0:
        # Spawn new barrier periodically
        new_barrier = spawner.spawn()
        barriers.append(new_barrier)
        gameobjects.append(new_barrier)
        last_barrier = new_barrier

    for barrier in barriers:
        barrier.move(-BARRIER_SPEED, 0)

    for horse in horses:
        horse.update_animation()

        if not horse.stopped:
            # Check collisions with barriers and bounds
            for barrier in barriers:
                if horse.rect.colliderect(barrier.rect):
                    horse.stop()
            if horse.rect.colliderect(upper_bound_rect) or horse.rect.colliderect(lower_bound_rect):
                horse.stop()
            horse.count_fitness()

    for object in gameobjects:
        object.draw(screen)

    current_fitnesses = [horse.fitness for horse in horses]
    current_fitness = max(current_fitnesses)
    
    if all(horse.stopped for horse in horses):
        # All horses stopped: evolve population and start new iteration
        UI.iteration_num += 1
        genecticAlg.learn([x.fitness for x in horses])

        local_best_fitness = max(horse.fitness for horse in horses)
        best_fitnesses.append(local_best_fitness)
        running_fitnesses.append(current_fitness)
        current_iteration += 1
        init_game()

    UI.draw(screen)
    UI.draw_marks(screen)

    # Display iteration and fitness info
    iteration_num_txt = font.render(f"Iteration: {current_iteration}", True, TEXT_COLOUR)
    current_fitness_txt = font.render(
        f"Current Fitness: {current_fitness}", True, TEXT_COLOUR)
    best_fitness_txt = font.render(
        f"Best Fitness: {best_fitnesses[-1] if best_fitnesses else 0}", True, TEXT_COLOUR)

    screen.blit(iteration_num_txt, (WIDTH / 2 - 90, HEIGHT + HUD_HEIGHT - 100))
    screen.blit(best_fitness_txt, (10, 5))
    screen.blit(current_fitness_txt, (10, 40))

    pygame.display.flip()
    pygame.time.Clock().tick(FRAME_RATE)
