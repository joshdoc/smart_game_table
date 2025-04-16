import pygame
import random
import sys
import time

# Constants
SCREEN_WIDTH = 1400 
SCREEN_HEIGHT = 1050
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUCK_RADIUS = 15
PADDLE_RADIUS = 30

# Simulated function for obtaining player positions
def cv_loop():
    """Simulate the cv_loop function to return random centroid positions."""
    # Replace this logic with actual centroid detection logic
    # Just for demonstration purposes
    x, y = pygame.mouse.get_pos()
    player1 =  [x, y]
    player2 = [SCREEN_WIDTH - PADDLE_RADIUS, random.randint(PADDLE_RADIUS, SCREEN_HEIGHT - PADDLE_RADIUS)]
    return player1, player2

class Puck(pygame.sprite.Sprite):
    def __init__(self, pos, rad, groups, obstacles):
        super().__init__(groups)
        self.image = pygame.Surface((PUCK_RADIUS * 2, PUCK_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (PUCK_RADIUS, PUCK_RADIUS), PUCK_RADIUS)
        #self.image.fill(WHITE)
        self.rect = self.image.get_rect(topleft = (640,360))

        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2((-1,1))
        self.speed = pygame.math.Vector2((200,200))
        self.old_rect = self.rect.copy()
        self.obstacles = obstacles

        self.mass = 20

    def collision(self,direction):
        collision_sprites = pygame.sprite.spritecollide(self, self.obstacles, False)
        if collision_sprites:
            if direction == 'horizontal':
                for sprite in collision_sprites:
					# collision on the right
                    if self.rect.right >= sprite.rect.left and self.old_rect.right <= sprite.old_rect.left:
                        self.rect.right = sprite.rect.left
                        self.pos.x = self.rect.x
                        self.direction[0] = -self.direction[0]
                        self.speed.x +=5

					# collision on the left
                    if self.rect.left <= sprite.rect.right and self.old_rect.left >= sprite.old_rect.right:
                        self.rect.left = sprite.rect.right
                        self.pos.x = self.rect.x
                        self.direction[0] = -self.direction[0]
                        self.speed.x +=5
		
            if direction == 'vertical':
                for sprite in collision_sprites:
                  # collision on the bottom
                    if self.rect.bottom >= sprite.rect.top and self.old_rect.bottom <= sprite.old_rect.top:
                        self.rect.bottom = sprite.rect.top
                        self.pos.y = self.rect.y
                        self.direction[1] = -self.direction[1]
                        self.speed.y +=5

                  # collision on the top
                    if self.rect.top <= sprite.rect.bottom and self.old_rect.top >= sprite.old_rect.bottom:
                        self.rect.top = sprite.rect.bottom
                        self.pos.y = self.rect.y
                        self.direction[1] = -self.direction[1]
                        self.speed.y +=5       
        
    def update(self,dt):
        self.old_rect = self.rect.copy() #previous  frame
        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()

        self.pos.x += self.direction.x * self.speed.x * dt
        self.rect.x = round(self.pos.x)
        self.collision('horizontal')
        self.pos.y += self.direction.y * self.speed.y * dt
        self.rect.y = round(self.pos.y)
        self.collision('vertical')
        
        # Bounce off top and bottom
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.direction[1] = -self.direction[1]
        if self.rect.left <= 0 or self.rect.right >= SCREEN_WIDTH:
            self.direction[0] = -self.direction[0]
    
class Paddle(pygame.sprite.Sprite):
    def __init__(self, init_pos, groups):
        super().__init__(groups)
        self.image = pygame.Surface((PADDLE_RADIUS * 2, PADDLE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (PADDLE_RADIUS, PADDLE_RADIUS), PADDLE_RADIUS)
        #self.image.fill(WHITE)
        self.rect = self.image.get_rect(topleft = init_pos)
        self.mass = 60
        self.velocity = pygame.math.Vector2((0,0))

    def update(self, dt):
        self.old_rect = self.rect.copy()
        old_pos = self.rect.center
        new_pos = cv_loop()[0]
        self.rect.center = new_pos
        # Calculate velocity based on position change
        self.velocity = [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]]

pygame.init()

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Air Hockey")

all_sprites = pygame.sprite.Group()
collision_sprites = pygame.sprite.Group()

# Create sprite groups
player1 = Paddle((PADDLE_RADIUS, SCREEN_HEIGHT // 2), [all_sprites,collision_sprites])
puck = Puck(30, (200,200), [all_sprites,collision_sprites], collision_sprites)
#player2 = Paddle((SCREEN_WIDTH - PADDLE_RADIUS, SCREEN_HEIGHT // 2), [all_sprites,collision_sprites])

# Main game loop
running = True

# Font for displaying text

font = pygame.font.SysFont("Arial", 24)

last_time = time.time()
while True:
    dt = time.time() - last_time
    last_time = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill(BLACK)
    all_sprites.update(dt)
    all_sprites.draw(screen)

    # Render Player 1's velocity and display it
    velocity_text = f"Velocity: {player1.velocity[0]}, {player1.velocity[1]}"
    velocity_surface = font.render(velocity_text, True, WHITE)
    screen.blit(velocity_surface, (10, 10))

    pygame.display.update()