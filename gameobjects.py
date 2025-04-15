import pygame
import random

WIDTH, HEIGHT = 800, 600
HUD_HEIGHT = 150
BARRIER_SEED = 20

class GameObject:
    x = 0
    y = 0

    def __init__(self, image_path, x=0, y=0, offset=(0, 0)):
        self.children = []
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.set_position(x, y)
        self.offset = offset

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        if len(self.children) == 0:
            return
        for child in self.children:
            child.draw(surface)

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.x = x
        self.y = y
        if len(self.children) == 0:
            return
        for child in self.children:
            child.set_position(x + child.offset[0], y + child.offset[1])

    def move(self, dx, dy):
        self.set_position(self.rect.x + dx, self.rect.y + dy)
        if len(self.children) == 0:
            return
        for child in self.children:
            self.set_position(self.rect.x + dx, self.rect.y + dy)

    def set_size(self, width, height):
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def scale_by(self, scale):
        self.image = pygame.transform.scale(self.image, (self.image.get_width() * scale, self.image.get_height() * scale))
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def get_position(self):
        return (self.x, self.y)

    def set_children(self, child):
        self.children.append(child)

    def recolor(self, color):
        self.image.fill(color, special_flags=pygame.BLEND_MULT)
        self.image.set_colorkey(None)

class Horse(GameObject):
    default_vacceleration = 0.2
    vspeed = 0
    vacceleration = 0
    stopped = False
    color = ()
    ribbon_fill = None
    ribbon_out = None

    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        self.images = []
        self.images.append(pygame.image.load("images/Horse_1.png"))
        self.images.append(pygame.image.load("images/Horse_2.png"))
        self.images.append(pygame.image.load("images/Horse_3.png"))
        self.animation_speed = 25
        self.frame_counter = 0
        self.current_frame = 0
        self.vspeed = 0
        self.vacceleration = 0
        self.stopped = False
        self.fitness = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.ribbon_fill = GameObject("images/Ribbon_fill.png", self.x, self.y, (16, -1))
        self.ribbon_fill.recolor(self.color)
        self.ribbon_out = GameObject("images/Ribbon_out.png", self.x, self.y, (14, -2))
        self.set_children(self.ribbon_fill)
        self.set_children(self.ribbon_out)

    def apply_vspeed(self):
        pos = self.get_position()
        self.set_position(pos[0], pos[1] + self.vspeed)

    def apply_vacceleration(self):
        self.vspeed += self.vacceleration

    def set_vspeed(self, speed):
        self.vspeed = speed

    def add_vspeed(self, delta):
        self.vspeed += delta

    def set_vacceleration(self, vacceleration):
        self.vacceleration = vacceleration

    def up(self):
        self.set_vacceleration(-self.default_vacceleration)

    def down(self):
        self.set_vacceleration(self.default_vacceleration)

    def stay(self):
        self.set_vacceleration(0)

    def stop(self):
        self.stopped = True
        self.set_vacceleration(0)
        self.set_vspeed(0)

    def draw(self, surface):
        super().draw(surface)

    def update_animation(self):
        self.frame_counter += 1
        if self.frame_counter >= self.animation_speed:
            self.frame_counter = 0
            self.current_frame = (self.current_frame + 1) % len(self.images)
            self.image = self.images[self.current_frame]

    def count_fitness(self):
        if self.stopped == False:
            self.fitness += 1
        if self.vspeed == 0:
            self.fitness -= 0.9

class Background(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)

class Barrier(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        self.scale_by(5)

class UI:
    iteration_num = 0
    horses = []
    horizontal_offset = 25
    vertical_offset = -25
    mark_size = 25

    def __init__(self, hud_image_path, x=0, y=HEIGHT):
        self.horses = []
        self.hud_image = pygame.image.load(hud_image_path)
        self.hud_rect = self.hud_image.get_rect(topleft=(x, y))

    def add_horses(self, children):
        new_horses = [child for child in children if child not in self.horses]
        self.horses.extend(new_horses)

    def draw_marks(self, screen):
        active_marks = [horse for horse in self.horses if not horse.stopped]
        unactive_marks = [horse for horse in self.horses if horse.stopped]
        width = 10
        xa = 0
        ya = HEIGHT + HUD_HEIGHT - self.mark_size
        for i, horse in enumerate(active_marks):
            pygame.draw.rect(screen, horse.color, (xa, ya, self.mark_size, self.mark_size))
            xa += self.horizontal_offset
            if (i + 1) % width == 0:
                ya += self.vertical_offset
                xa = 0
        xu = WIDTH - self.mark_size
        yu = HEIGHT + HUD_HEIGHT - self.mark_size
        for i, horse in enumerate(unactive_marks):
            pygame.draw.rect(screen, horse.color, (xu, yu, self.mark_size, self.mark_size))
            xu -= self.horizontal_offset
            if (i + 1) % width == 0:
                yu += self.vertical_offset
                xu = WIDTH - self.mark_size

    def draw(self, surface):
        surface.blit(self.hud_image, self.hud_rect)

class Spawner:
    active = True
    delay = 500
    tick_counter = 0

    def __init__(self, barrier_image_path, delay):
        self.barrier_image_path = barrier_image_path
        self.delay = delay

    def handle(self):
        if self.active:
            self.tick_counter += 1
            if self.tick_counter >= self.delay:
                self.spawn()
                self.tick_counter = 0

    def spawn(self):
        random_y = random.randint(0, HEIGHT - 100)
        new_barrier = Barrier(self.barrier_image_path, WIDTH, random_y)
        return new_barrier
