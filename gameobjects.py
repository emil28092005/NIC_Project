import pygame
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
        self.rect.x = x - self.image.get_width() / 2
        self.rect.y = y - self.image.get_height() / 2
        self.x = x - self.image.get_width() / 2
        self.y = y - self.image.get_width() / 2
    def move(self, dx, dy):
        self.set_position(self.rect.x + dx, self.rect.y + dy)
    def set_size(self, width, height):
        self.image = pygame.transform.scale(self.image, (width,height))
        self.rect = self.image.get_rect(topleft=(self.x,self.y))
    def scale_by(self, scale):
        self.image = pygame.transform.scale(self.image, (self.image.get_width() * scale, self.image.get_height() * scale))
        self.rect = self.image.get_rect(topleft=(self.x,self.y))
    
    
        
class Horse(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)

class Background(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)

class Barrier(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        