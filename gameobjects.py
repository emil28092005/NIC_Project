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
    default_vacceleration = 1
    vspeed = 0
    vacceleration = 0
    def __init__(self, image_path, x, y):
        self.vspeed = 0
        self.vacceleration = 0
        super().__init__(image_path, x, y)
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




class Background(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)

class Barrier(GameObject):
    def __init__(self, image_path, x, y):
        super().__init__(image_path, x, y)
        