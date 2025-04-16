import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1050
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PUCK_RADIUS = 15
PADDLE_RADIUS = 30

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Air Hockey")

class Puck(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PUCK_RADIUS * 2, PUCK_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (PUCK_RADIUS, PUCK_RADIUS), PUCK_RADIUS)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.velocity = [random.choice([-4, 4]), random.choice([-4, 4])]
        self.mask = pygame.mask.from_surface(self.image)
        
    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

        # Bounce off top and bottom
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.velocity[1] = -self.velocity[1]
        
class Paddle(pygame.sprite.Sprite):
    def __init__(self, init_pos):
        super().__init__()
        self.image = pygame.Surface((PADDLE_RADIUS * 2, PADDLE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (PADDLE_RADIUS, PADDLE_RADIUS), PADDLE_RADIUS)
        self.rect = self.image.get_rect(center=init_pos)
        self.mask = pygame.mask.from_surface(self.image)

    def update_position(self, new_pos):
        old_pos = self.rect.center
        self.rect.center = new_pos
        # Calculate velocity based on position change
        self.velocity = [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]]

# Create sprite groups
puck = Puck()
player1 = Paddle((PADDLE_RADIUS, SCREEN_HEIGHT // 2))
player2 = Paddle((SCREEN_WIDTH - PADDLE_RADIUS, SCREEN_HEIGHT // 2))

all_sprites = pygame.sprite.Group(puck, player1, player2)

# Simulated function for obtaining player positions
def cv_loop():
    """Simulate the cv_loop function to return random centroid positions."""
    # Replace this logic with actual centroid detection logic
    # Just for demonstration purposes
    x, y = pygame.mouse.get_pos()
    player1 =  [x, y]
    player2 = [SCREEN_WIDTH - PADDLE_RADIUS, random.randint(PADDLE_RADIUS, SCREEN_HEIGHT - PADDLE_RADIUS)]
    return player1, player2

# Function to clamp paddle positions within screen boundaries
def clamp_paddle(pos):
    pos[0] = max(PADDLE_RADIUS, min(SCREEN_WIDTH - PADDLE_RADIUS, pos[0]))
    pos[1] = max(PADDLE_RADIUS, min(SCREEN_HEIGHT - PADDLE_RADIUS, pos[1]))

# Main game loop
running = True

# Font for displaying text
font = pygame.font.SysFont("Arial", 24)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the positions of the paddles from the cv_loop function
    player1_pos, player2_pos = cv_loop()
    player1.update_position(player1_pos)
    player2.update_position(player2_pos)

  
    # Update puck position
    puck.update()

    # Check for collisions using sprite masks
    if pygame.sprite.collide_circle(puck, player1) or pygame.sprite.collide_circle(puck, player2):
        puck.velocity[0] = -puck.velocity[0]
    
    # Check for scoring (left and right)
    if puck.rect.left <= 0 or puck.rect.right >= SCREEN_WIDTH:
        # Reset puck position and velocity
        puck.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        puck.velocity = [random.choice([-4, 4]), random.choice([-4, 4])]
        
    # Clear screen
    screen.fill(BLACK)

    # Render Player 1's velocity and display it
    velocity_text = f"Velocity: {player1.velocity[0]}, {player1.velocity[1]}"
    velocity_surface = font.render(velocity_text, True, WHITE)
    screen.blit(velocity_surface, (10, 10))
    '''score_text = font.render(f"{score1} : {score2}", True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH // 2 - 50, 10))'''
    
    # Draw all sprites
    all_sprites.draw(screen)

    # Update display
    pygame.display.update()
    pygame.time.Clock().tick(30)

pygame.quit()
sys.exit()