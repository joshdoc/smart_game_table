####################################################################################################
# hockey.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                            #
#                                                                                                  #
# This is the hockey demo.                                                                         #
#                                                                                                  #
####################################################################################################

####################################################################################################
# Imports                                                                                          #
####################################################################################################


import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

import cv
from sgt_types import DetectedCentroids, Centroid, Loop_Result_t

####################################################################################################
# Types                                                                                            #
####################################################################################################


@dataclass
class Vector:
    x: float
    y: float

    def __init__(self, x: int = 0, y: int = 0, vec: tuple[float, float] = (0, 0)) -> None:
        if vec != (0, 0):
            self.x = vec[0]
            self.y = vec[1]
        else:
            self.x = x
            self.y = y


####################################################################################################
# Constants                                                                                        #
####################################################################################################


SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1050

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255,0,0)

PUCK_RADIUS = 15*2
PADDLE_RADIUS = 130

VELOCITY_SCALE = 2
FRICTION = 0.99

####################################################################################################
# Globals                                                                                          #
####################################################################################################

all_sprites = pygame.sprite.Group()
collision_sprites = pygame.sprite.Group()
clock = pygame.time.Clock()
prev_time = time.time()

####################################################################################################
# Classes                                                                                          #
####################################################################################################


class Puck(pygame.sprite.Sprite):
    def __init__(
        self,
        groups: Any,
        obstacles: Any,
        pos: tuple[int, int] = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
        radius: int = PUCK_RADIUS,
    ):
        super().__init__(groups)
        self.obstacles = obstacles

        self.radius = radius

        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (radius, radius), radius)

        self.rect = self.image.get_rect(center=pos)
        self.prev_rect = self.rect

        self.velocity: Vector = Vector(0, 0)

    def _collision(self) -> None:
        collision_sprites = pygame.sprite.spritecollide(self, self.obstacles, False)
        for sprite in collision_sprites:
            if self.rect.right >= sprite.rect.left and self.prev_rect.right <= sprite.prev_rect.left:
                self.rect.right = sprite.rect.left
            if self.rect.left <= sprite.rect.right and self.prev_rect.left >= sprite.prev_rect.right:
                self.rect.left = sprite.rect.right
            if self.rect.bottom >= sprite.rect.top and self.prev_rect.bottom <= sprite.prev_rect.top:
                self.rect.bottom = sprite.rect.top
            if self.rect.top <= sprite.rect.bottom and self.prev_rect.top >= sprite.prev_rect.bottom:
                self.rect.top = sprite.rect.bottom

            self.velocity.x = sprite.velocity.x / VELOCITY_SCALE
            self.velocity.y = sprite.velocity.y / VELOCITY_SCALE

    def _bounds_check(self):
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

        if self.rect.top <= 0:
            self.rect.top = 0

        if self.rect.left <= 0:
            self.rect.left = 0

        if self.rect.right >= SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

        if self.rect.left == 0 or self.rect.right == SCREEN_WIDTH:
            self.velocity.x = -self.velocity.x

        if self.rect.top == 0 or self.rect.bottom == SCREEN_HEIGHT:
            self.velocity.y = -self.velocity.y

    def update(self, dt: float) -> None:
        self.prev_rect = self.rect.copy()

        self.rect.x += int(self.velocity.x * dt)
        self.rect.y += int(self.velocity.y * dt)
        
        self._collision()
        self._bounds_check()

        self.velocity.x *= FRICTION
        self.velocity.y *= FRICTION


class Paddle(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int], groups: Any, radius=PADDLE_RADIUS):
        super().__init__(groups)

        self.radius = radius

        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (radius, radius), radius)

        self.rect = self.image.get_rect(center=pos)
        self.prev_rect: self.rect

        self.velocity: Vector = Vector(0, 0)

    def update(self, dt, centroid: cv.Centroid = cv.Centroid(0, 0, np.zeros(1))):
        self.prev_rect = self.rect.copy()
        self.rect.center = (centroid.xpos, centroid.ypos)

        if dt > 0:
            self.velocity.x = (self.rect.x - self.prev_rect.x) / dt
            self.velocity.y = (self.rect.y - self.prev_rect.y) / dt
        else:
            self.velocity.x = 0
            self.velocity.y = 0


####################################################################################################
# Public Functions                                                                                 #
####################################################################################################


def init(_) -> None:
    global screen, font, puck, player1, player2

    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.SysFont("Arial", 24)

    puck = Puck([all_sprites], collision_sprites)
    player1 = Paddle((PADDLE_RADIUS, SCREEN_HEIGHT // 2), [all_sprites, collision_sprites])
    player2 = Paddle(
        (SCREEN_WIDTH - PADDLE_RADIUS, SCREEN_HEIGHT // 2), [all_sprites, collision_sprites]
    )


def loop(centroids: DetectedCentroids, dt: float) -> None:
    global prev_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill(BLACK)
    puck.update(dt)

    paddle_locations: list[Centroid] = centroids.cds
    if len(paddle_locations) == 2:
        player1.update(dt, paddle_locations[0])
        player2.update(dt, paddle_locations[1])

    all_sprites.draw(screen)

    velocity_text = f"Paddle 1 Velocity: {player1.velocity.x:.2f}, {player1.velocity.y:.2f}"
    velocity_surface = font.render(velocity_text, True, WHITE)
    screen.blit(velocity_surface, (10, 10))

    velocity_text = f"Puck Speed: {puck.velocity.x:.2f}, {puck.velocity.y:.2f}"
    velocity_surface = font.render(velocity_text, True, WHITE)
    screen.blit(velocity_surface, (10, 36))

    framerate = clock.get_fps()
    framerate_text = font.render(f"FPS: {framerate:.2f}", True, WHITE)
    screen.blit(framerate_text, (10, 900))
    clock.tick(30)

    pygame.display.update()

def deinit() -> None:
    pygame.quit()


####################################################################################################
# Main                                                                                             #
####################################################################################################


def main() -> None:
    cv.cv_init(detect_fingers=True, detect_cds=True)
    init()

    loop_res: Loop_Result_t = Loop_Result_t.CONTINUE

    while loop_res == Loop_Result_t.CONTINUE:
        centroids = cv.cv_loop()
        loop_res = loop()
    
    deinit()

if __name__ == "__main__":
    main()
