import pygame
import math

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Differential Drive Robot Simulation")

background = pygame.image.load('images/background.png')
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y, width=20, height=20, speed=5, turn_speed=0.05):
        super().__init__()
        
        self.width = width
        self.height = height
        self.speed = speed
        self.turn_speed = turn_speed
        self.x = x
        self.y = y
        self.angle = 0

        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height))
        
    def update(self, keys):
        if keys[pygame.K_UP]:
            self.x += self.speed * math.cos(self.angle)
            self.y += self.speed * math.sin(self.angle)
        if keys[pygame.K_DOWN]:
            self.x -= self.speed * math.cos(self.angle)
            self.y -= self.speed * math.sin(self.angle)
        if keys[pygame.K_LEFT]:
            self.angle -= self.turn_speed
        if keys[pygame.K_RIGHT]:
            self.angle += self.turn_speed
        
    def draw(self, surface):
        self.rect.center = (self.x, self.y)
        self.image.fill((0, 0, 255))
        rotated_image = pygame.transform.rotate(self.image, -math.degrees(self.angle))
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        surface.blit(rotated_image, rotated_rect.topleft)

clock = pygame.time.Clock()
running = True

robot = Robot(WIDTH // 2, HEIGHT // 2)


while running:
    screen.fill((255, 255, 255)) 
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    robot.update(keys)
    robot.draw(screen)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()