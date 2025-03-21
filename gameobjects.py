import pygame
import random

WIDTH, HEIGHT = 800, 600

class GameObject:
    x = 0
    y = 0
    
    def __init__(self, image_path, x=0, y=0, offset = (0, 0)):
        self.children = []
        self.image = pygame.image.load(image_path)  # Load the image
        self.rect = self.image.get_rect(topleft=(x,y))  # Set the rectangle for the image
        self.set_position(x, y)  # Set initial position
        self.offset = offset
    def draw(self, surface):
        surface.blit(self.image, self.rect)  # Draw the image on the surface
        if len(self.children) == 0: return
        for child in self.children:
            child.draw(surface)
    def set_position(self, x, y):
        self.rect.x = x  # Update rectangle position
        self.rect.y = y
        self.x = x  # Update object position
        self.y = y
        if len(self.children) == 0: return
        for child in self.children:
            child.set_position(x + child.offset[0], y + child.offset[1])
    def move(self, dx, dy):
        self.set_position(self.rect.x + dx, self.rect.y + dy)  # Move object by dx and dy
        if len(self.children) == 0: return
        for child in self.children:
            self.set_position(self.rect.x + dx, self.rect.y + dy)
    def set_size(self, width, height):
        self.image = pygame.transform.scale(self.image, (width,height))  # Resize the image
        self.rect = self.image.get_rect(topleft=(self.x,self.y))  # Update rectangle size
    def scale_by(self, scale):
        self.image = pygame.transform.scale(self.image, (self.image.get_width() * scale, self.image.get_height() * scale))  # Scale the image
        self.rect = self.image.get_rect(topleft=(self.x,self.y))  # Update rectangle size
    def get_position(self):
        return (self.x, self.y)  # Return current position
    def set_children(self, child):
        self.children.append(child)
    def recolor(self, color): #Method to recolor
        self.image.fill(color, special_flags=pygame.BLEND_MULT)
        self.image.set_colorkey(None)  # Explicitly disable colorkey

class Horse(GameObject):
    default_vacceleration = 0.5  # Default acceleration value
    vspeed = 0  # Vertical speed
    vacceleration = 0  # Vertical acceleration
    stopped = False  # Whether the horse is stopped
    color = ()  # Color of the horse's mark
    ribbon_fill = None
    ribbon_out = None
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        self.vspeed = 0
        self.vacceleration = 0
        self.stopped = False
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color for the mark
        self.ribbon_fill = GameObject("images/Ribbon_fill.png", self.x, self.y, (16, -1))
        self.ribbon_fill.recolor(self.color)
        self.ribbon_out = GameObject("images/Ribbon_out.png", self.x, self.y, (14, -2))
        self.set_children(self.ribbon_fill)
        self.set_children(self.ribbon_out)
        # self.set_size(75, 75)  # Set the size of the horse
    def apply_vspeed(self):
        pos = self.get_position()
        self.set_position(pos[0], pos[1] + self.vspeed)  # Apply vertical speed to position
    def apply_vacceleration(self):
        self.vspeed += self.vacceleration  # Apply acceleration to speed
    def set_vspeed(self, speed):
        self.vspeed = speed  # Set vertical speed
    def add_vspeed(self, delta):
        self.vspeed += delta  # Add to vertical speed
    def set_vacceleration(self, vacceleration):
        self.vacceleration = vacceleration  # Set vertical acceleration
    def up(self):
        self.set_vacceleration(-self.default_vacceleration)  # Move up
    def down(self):
        self.set_vacceleration(self.default_vacceleration)  # Move down
    def stay(self):
        self.set_vacceleration(0)  # Stop vertical movement
    def stop(self):
        self.stopped = True  # Stop the horse
        self.set_vacceleration(0)
        self.set_vspeed(0)
    def draw(self, surface):
        super().draw(surface)  # Draw the horse
        #pygame.draw.rect(surface, self.color, (self.x + 40, self.y + 20, 25, 25))  # Draw the horse's mark

class Background(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)  # Initialize background

class Barrier(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)  # Initialize barrier
        self.scale_by(5)

class UI:
    horses = []  # List of child objects
    horizontal_offset = 25  # Horizontal spacing between marks
    vertical_offset = -25  # Vertical spacing between marks
    mark_size = 25  # Size of the marks
    def __init__(self):
        self.horses = []  # Initialize children list
    def add_horses(self, children):
        new_horses = [child for child in children if child not in self.horses]  # Add new children
        self.horses.extend(new_horses)
    def draw_marks(self, screen):
        active_marks = []  # List of active horses
        unactive_marks = []  # List of inactive horses
        for i in range(len(self.horses)):
            if isinstance(self.horses[i], Horse):
                if (self.horses[i].stopped == False):
                    active_marks.append(self.horses[i])  # Add active horses
                else:
                    unactive_marks.append(self.horses[i])  # Add inactive horses

        width = 10  # Number of marks per row
        active_marks_section_pos = (0 - self.mark_size, 600 - self.mark_size)  # Starting position for active marks
        xa = active_marks_section_pos[0]
        ya = active_marks_section_pos[1]
        for i in range(len(active_marks)):
            xa += self.horizontal_offset
            pygame.draw.rect(screen, self.horses[i].color, (xa, ya, self.mark_size, self.mark_size))  # Draw active marks
            if xa >= width * (self.mark_size - 3):  # Move to the next row if the row is full
                ya += self.vertical_offset
                xa = active_marks_section_pos[0]

        unactive_marks_section_pos = (800, 600 - self.mark_size)  # Starting position for inactive marks
        xu = unactive_marks_section_pos[0]
        yu = unactive_marks_section_pos[1]
        for i in range(len(unactive_marks)):
            xu -= self.horizontal_offset
            pygame.draw.rect(screen, unactive_marks[i].color, (xu, yu, self.mark_size, self.mark_size))  # Draw inactive marks
            if 800 - xu >= width * self.mark_size:  # Move to the next row if the row is full
                yu += self.vertical_offset
                xu = unactive_marks_section_pos[0]


class Spawner:
    active = True  # Whether the spawner is active
    frequency = 500  # Number of ticks between spawns
    tick_counter = 0  # Counter for ticks

    def __init__(self, barrier_image_path):
        self.barrier_image_path = barrier_image_path  # Path to the barrier image

    def handle(self):
        if self.active:
            self.tick_counter += 1  # Increment tick counter
            if self.tick_counter >= self.frequency:  # Check if it's time to spawn
                self.spawn()
                self.tick_counter = 0  # Reset tick counter

    def spawn(self):
        random_y = random.randint(0, HEIGHT - 100)  # Random Y position for the barrier
        new_barrier = Barrier(self.barrier_image_path, WIDTH, random_y)  # Create a new barrier
        return new_barrier
