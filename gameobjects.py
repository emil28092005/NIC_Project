import pygame
import random

WIDTH, HEIGHT = 800, 600
HUD_HEIGHT = 150
BARRIER_SEED = 20

class GameObject:
    """
    Base class for all game objects.
    Handles image loading, positioning, drawing, and child objects.
    """
    x = 0
    y = 0

    def __init__(self, image_path, x=0, y=0, offset=(0, 0)):
        self.children = []
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.set_position(x, y)
        self.offset = offset

    def draw(self, surface):
        # Draw self and all children recursively
        surface.blit(self.image, self.rect)
        if not self.children: 
            return
        for child in self.children:
            child.draw(surface)

    def set_position(self, x, y):
        # Set position and update children positions with offsets
        self.rect.x = x
        self.rect.y = y
        self.x = x
        self.y = y
        if not self.children: 
            return
        for child in self.children:
            child.set_position(x + child.offset[0], y + child.offset[1])

    def move(self, dx, dy):
        # Move object by delta and update children accordingly
        self.set_position(self.rect.x + dx, self.rect.y + dy)
        if not self.children: 
            return
        for child in self.children:
            self.set_position(self.rect.x + dx, self.rect.y + dy)

    def set_size(self, width, height):
        # Resize image and update rect
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def scale_by(self, scale):
        # Scale image by a factor
        new_size = (self.image.get_width() * scale, self.image.get_height() * scale)
        self.image = pygame.transform.scale(self.image, new_size)
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def get_position(self):
        return (self.x, self.y)

    def set_children(self, child):
        # Add a child GameObject
        self.children.append(child)

    def recolor(self, color):
        # Apply color tint to image
        self.image.fill(color, special_flags=pygame.BLEND_MULT)
        self.image.set_colorkey(None)

class Horse(GameObject):
    """
    Represents a horse character with vertical movement, animation, and fitness tracking.
    Supports acceleration, speed, and stopping.
    """
    default_vacceleration = 0.2

    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        # Load animation frames
        self.images = [
            pygame.image.load("images/Horse_1.png"),
            pygame.image.load("images/Horse_2.png"),
            pygame.image.load("images/Horse_3.png")
        ]
        self.animation_speed = 25
        self.frame_counter = 0
        self.current_frame = 0
        self.vspeed = 0
        self.vacceleration = 0
        self.stopped = False
        self.fitness = 0
        # Assign random color for ribbons
        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.ribbon_fill = GameObject("images/Ribbon_fill.png", self.x, self.y, (16, -1))
        self.ribbon_fill.recolor(self.color)
        self.ribbon_out = GameObject("images/Ribbon_out.png", self.x, self.y, (14, -2))
        self.set_children(self.ribbon_fill)
        self.set_children(self.ribbon_out)

    def apply_vspeed(self):
        # Move vertically by current speed
        self.set_position(self.x, self.y + self.vspeed)

    def apply_vacceleration(self):
        # Update speed by acceleration
        self.vspeed += self.vacceleration

    def set_vspeed(self, speed):
        self.vspeed = speed

    def add_vspeed(self, delta):
        self.vspeed += delta

    def set_vacceleration(self, vacceleration):
        self.vacceleration = vacceleration

    def up(self):
        # Accelerate upward
        self.set_vacceleration(-self.default_vacceleration)

    def down(self):
        # Accelerate downward
        self.set_vacceleration(self.default_vacceleration)

    def stay(self):
        # No vertical acceleration
        self.set_vacceleration(0)

    def stop(self):
        # Stop all movement
        self.stopped = True
        self.set_vacceleration(0)
        self.set_vspeed(0)

    def update_animation(self):
        # Animate horse by cycling frames
        self.frame_counter += 1
        if self.frame_counter >= self.animation_speed:
            self.frame_counter = 0
            self.current_frame = (self.current_frame + 1) % len(self.images)
            self.image = self.images[self.current_frame]

    def count_fitness(self):
        # Increase fitness if moving; penalize if speed zero
        if not self.stopped:
            self.fitness += 1
        if self.vspeed == 0:
            self.fitness -= 0.9

class Background(GameObject):
    """
    Background image for the game scene.
    """
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)

class Barrier(GameObject):
    """
    Obstacles that move horizontally and interact with horses.
    """
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        self.scale_by(5)  # Make barrier larger for visibility

class UI:
    """
    Handles HUD display and horse status markers.
    """
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
        # Add horses to UI tracking list
        self.horses.extend([child for child in children if child not in self.horses])

    def draw_marks(self, screen):
        # Draw colored squares representing active and stopped horses
        active = [h for h in self.horses if not h.stopped]
        inactive = [h for h in self.horses if h.stopped]
        
        xa, ya = 0, HEIGHT + HUD_HEIGHT - self.mark_size
        for i, horse in enumerate(active):
            pygame.draw.rect(screen, horse.color, (xa, ya, self.mark_size, self.mark_size))
            xa += self.horizontal_offset
            if (i+1) % 10 == 0:
                ya += self.vertical_offset
                xa = 0
        
        xu, yu = WIDTH - self.mark_size, HEIGHT + HUD_HEIGHT - self.mark_size
        for i, horse in enumerate(inactive):
            pygame.draw.rect(screen, horse.color, (xu, yu, self.mark_size, self.mark_size))
            xu -= self.horizontal_offset
            if (i+1) % 10 == 0:
                yu += self.vertical_offset
                xu = WIDTH - self.mark_size

    def draw(self, surface):
        # Draw HUD image
        surface.blit(self.hud_image, self.hud_rect)

class Spawner:
    """
    Spawns barriers at intervals to challenge horses.
    """
    active = True
    delay = 500
    tick_counter = 0

    def __init__(self, barrier_image_path, delay):
        self.barrier_image_path = barrier_image_path
        self.delay = delay

    def handle(self):
        # Increment tick counter and spawn barrier when delay reached
        if self.active:
            self.tick_counter += 1
            if self.tick_counter >= self.delay:
                self.spawn()
                self.tick_counter = 0

    def spawn(self):
        # Create a new barrier at a random vertical position on the right edge
        return Barrier(self.barrier_image_path, WIDTH, random.randint(0, HEIGHT - 100))
