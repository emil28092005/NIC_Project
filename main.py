import pygame
import sys
from gameobjects import *
from genetic_alg import GeneticAlgorithm
import matplotlib.pyplot as plt
import random

POPULATION_SIZE = 500
MUTATION_RATE = 0.5
POPULATION_NEW = 0.1
POPULATION_BEST = 0.3
FRAME_RATE = 300
BARRIER_SPEED = 10
BARRIER_DELAY = 100
MAX_ITERATIONS = -1
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
    random.seed(BARRIER_SEED)
    grass = Background("images/Grass.jpg", 0, 0)
    grass.set_size(WIDTH, HEIGHT)

    gameobjects = [grass]
    barriers = []

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

    upper_bound_rect = pygame.Rect(0, 0, WIDTH, -10)
    lower_bound_rect = pygame.Rect(0, HEIGHT, WIDTH, 10)

    last_barrier = barriers[0]

    UI.add_horses(horses)


def get_features(horse: Horse):
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
            plt.plot(running_fitnesses)
            plt.xlabel("Iteration")
            plt.ylabel("Current Fitness")
            plt.title("Current Fitness per Iteration")
            plt.show()
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()

    for i, horse in enumerate(horses):
        if not horse.stopped:
            data = get_features(horse)
            res = genecticAlg.predict(data, i)
            if res == 0:
                horse.up()
            elif res == 1:
                horse.down()
            else:
                horse.stay()
        else:
            horse.move(-BARRIER_SPEED, 0)

        horse.apply_vacceleration()
        horse.apply_vspeed()
        horse.draw(screen)

    spawner.handle()
    if spawner.tick_counter == 0:
        new_barrier = spawner.spawn()
        barriers.append(new_barrier)
        gameobjects.append(new_barrier)
        last_barrier = new_barrier

    for barrier in barriers:
        barrier.move(-BARRIER_SPEED, 0)

    for horse in horses:
        horse.update_animation()

        if not horse.stopped:
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
        UI.iteration_num += 1
        genecticAlg.learn([x.fitness for x in horses])

        local_best_fitness = max(horse.fitness for horse in horses)
        best_fitnesses.append(local_best_fitness)
        running_fitnesses.append(current_fitness)
        current_iteration += 1
        init_game()

    UI.draw(screen)
    UI.draw_marks(screen)

    iteration_num_txt = font.render(f"iteration: {current_iteration}", True, BLACK)
    current_fitness_txt = font.render(
        f"Current Fitness: {current_fitness}", True, BLACK)
    best_fitness_txt = font.render(
        f"Best Fitness: {best_fitnesses[-1] if best_fitnesses else 0}", True, BLACK)

    screen.blit(iteration_num_txt, (WIDTH / 2 - 90, HEIGHT + HUD_HEIGHT - 120))
    screen.blit(best_fitness_txt, (WIDTH / 2 - 120, HEIGHT + HUD_HEIGHT - 80))
    screen.blit(current_fitness_txt, (WIDTH / 2 - 140, HEIGHT + HUD_HEIGHT - 40))

    pygame.display.flip()
    pygame.time.Clock().tick(FRAME_RATE)
