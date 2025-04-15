import pygame
import sys
import random

#from cv import cv_init, cv_loop, update_contours, nothing
from cv import *

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

# Position and velocity variables
puck_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
puck_vel = [random.choice([-4, 4]), random.choice([-4, 4])]

# Initialize player positions
player1_pos = [PADDLE_RADIUS, SCREEN_HEIGHT // 2]
player2_pos = [SCREEN_WIDTH - PADDLE_RADIUS, SCREEN_HEIGHT // 2]

score1 = 0
score2 = 0

cv_init()

# Function to clamp paddle positions within screen boundaries
def clamp_paddle(pos):
    pos[0] = max(PADDLE_RADIUS, min(SCREEN_WIDTH - PADDLE_RADIUS, pos[0]))
    pos[1] = max(PADDLE_RADIUS, min(SCREEN_HEIGHT - PADDLE_RADIUS, pos[1]))

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the positions of the paddles from the cv_loop function
    player1_pos, player2_pos = cv_loop()

    # Ensure paddles stay on the screen
    clamp_paddle(player1_pos)
    clamp_paddle(player2_pos)

    # Update puck position
    puck_pos[0] += puck_vel[0]
    puck_pos[1] += puck_vel[1]

    # Wall collision (top and bottom)
    if puck_pos[1] <= PUCK_RADIUS or puck_pos[1] >= SCREEN_HEIGHT - PUCK_RADIUS:
        puck_vel[1] = -puck_vel[1]

    # Scoring
    if puck_pos[0] <= PUCK_RADIUS:
        score2 += 1
        puck_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        puck_vel = [random.choice([-4, 4]), random.choice([-4, 4])]
    elif puck_pos[0] >= SCREEN_WIDTH - PUCK_RADIUS:
        score1 += 1
        puck_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        puck_vel = [random.choice([-4, 4]), random.choice([-4, 4])]

    # Paddle collision
    for paddle_pos in [player1_pos, player2_pos]:
        dist = ((puck_pos[0] - paddle_pos[0]) ** 2 + (puck_pos[1] - paddle_pos[1]) ** 2) ** 0.5
        if dist <= PUCK_RADIUS + PADDLE_RADIUS:
            norm = [puck_pos[0] - paddle_pos[0], puck_pos[1] - paddle_pos[1]]
            norm_magnitude = (norm[0] ** 2 + norm[1] ** 2) ** 0.5
            norm = [norm[0] / norm_magnitude, norm[1] / norm_magnitude]
            relative_velocity = puck_vel[0] * norm[0] + puck_vel[1] * norm[1]
            puck_vel[0] -= 2 * relative_velocity * norm[0]
            puck_vel[1] -= 2 * relative_velocity * norm[1]

    # Drawing
    screen.fill(BLACK)
    pygame.draw.circle(screen, WHITE, puck_pos, PUCK_RADIUS)
    pygame.draw.circle(screen, WHITE, player1_pos, PADDLE_RADIUS)
    pygame.draw.circle(screen, WHITE, player2_pos, PADDLE_RADIUS)

    # Display scores
    font = pygame.font.Font(None, 74)
    score_text = font.render(f"{score1} : {score2}", True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH // 2 - 50, 10))

    # Update display
    pygame.display.flip()
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()