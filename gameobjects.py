import pygame
import random

class GameObject:
    x = 0
    y = 0
    def __init__(self, image_path, x=0, y=0):
        self.image = pygame.image.load(image_path)
        self.rect = self.image.get_rect(topleft=(x,y))
        self.set_position(x, y)
    def draw(self, surface):
        surface.blit(self.image, self.rect)
    def set_position(self, x, y): # TODO: FIX CENTERING
        self.rect.x = x
        self.rect.y = y
        self.x = x
        self.y = y
    def move(self, dx, dy):
        self.set_position(self.rect.x + dx, self.rect.y + dy)
    def set_size(self, width, height):
        self.image = pygame.transform.scale(self.image, (width,height))
        self.rect = self.image.get_rect(topleft=(self.x,self.y))
    def scale_by(self, scale):
        self.image = pygame.transform.scale(self.image, (self.image.get_width() * scale, self.image.get_height() * scale))
        self.rect = self.image.get_rect(topleft=(self.x,self.y))
    def get_position(self):
        return (self.x, self.y)
    
    
        
class Horse(GameObject):
    default_vacceleration = 0.5
    vspeed = 0
    vacceleration = 0
    stopped = False
    color = ()
    def __init__(self, image_path, x, y):
        self.vspeed = 0
        self.vacceleration = 0
        self.stopped = False
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        super().__init__(image_path, x, y)
        self.set_size(75, 75)
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




class Background(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)

class Barrier(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        
class UI:
    children = []
    horizontal_offset = 25
    mark_size = 5
    def __init__(self):
        self.children = []
    def add_children(self, children):
        new_children = [child for child in children if child not in self.children]
        self.children.extend(new_children)
    def draw_marks(self, screen):
        x0 = 50
        
        y0 = 300
        
        for child in self.children:
            if isinstance(child, Horse):
                x0 += self.horizontal_offset
                x1 = x0 + self.mark_size
                y1 = y0 - self.mark_size
                pygame.draw.rect(screen, child.color, (x0, y0, x1, y1))
                