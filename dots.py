####################################################################################################
# CV.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                                #
#                                                                                                  #
# This file runs the user test game for a set duration and dot size without collecting any data.   #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

import math
import random
import time

import numpy as np
import pygame

import cv
from sgt_types import DetectedCentroids, Loop_Result_t

####################################################################################################
# CONSTANTS                                                                                        #
####################################################################################################


WIDTH, HEIGHT = 1400, 1050
TARGET_SPAWN_RATE = 0.3

MAX_TARGETS = 10
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

TARGET_RADIUS = 20
CENTROID_RADIUS = 15
TARGET_SPACING = TARGET_RADIUS * 2
GAME_DURATION = 30


####################################################################################################
# GLOBALS                                                                                          #
####################################################################################################


####################################################################################################
# LOCAL FUNCTIONS                                                                                  #
####################################################################################################


# Function to calculate distance between two points
def _distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


####################################################################################################
# GLOBAL FUNCTIONS                                                                                 #
####################################################################################################


def init(_=None):
    global clock, screen, font, last_spawn_time, start_time, score, targets
    pygame.init()

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((1400, 1050))
    pygame.display.set_caption("Tap the Target Game")

    font = pygame.font.SysFont("Arial", 24)

    targets = []  # List of targets
    score = 0
    last_spawn_time = time.time()
    start_time = time.time()


def loop(centroids: DetectedCentroids, _=None) -> Loop_Result_t:
    global score, last_spawn_time, targets
    retVal = Loop_Result_t.CONTINUE

    screen.fill(WHITE)

    # End the game if the time limit is reached or escape sequence is pressed
    elapsed_time = time.time() - start_time
    if centroids.escape or elapsed_time >= GAME_DURATION:
        retVal = Loop_Result_t.EXIT

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            retVal = Loop_Result_t.EXIT

    for target in targets[:]:
        target_x, target_y = target

        for centroid in centroids.fingers:
            pygame.draw.circle(
                screen, BLUE, (centroid.xpos, centroid.ypos), CENTROID_RADIUS
            )

            distance_from_center = _distance(
                centroid.xpos, centroid.ypos, target_x, target_y
            )

            # Check if click is within the target
            if distance_from_center <= TARGET_RADIUS + CENTROID_RADIUS:
                score += 1
                if target in targets:
                    targets.remove(target)

    # Spawn new targets at the defined spawn rate
    if len(targets) < MAX_TARGETS and (
        time.time() - last_spawn_time >= TARGET_SPAWN_RATE
    ):
        last_spawn_time = time.time()
        created = False

        while not created:
            created = True
            target_pos = (
                random.randint(TARGET_RADIUS, WIDTH - TARGET_RADIUS),
                random.randint(TARGET_RADIUS, HEIGHT - TARGET_RADIUS),
            )

            for target in targets:
                if (
                    _distance(target_pos[0], target_pos[1], target[0], target[1])
                    < TARGET_SPACING
                ):
                    created = False
                    break

            if created:
                targets.append((target_pos[0], target_pos[1]))

    # Draw all targets
    for target in targets:
        pygame.draw.circle(screen, RED, (target[0], target[1]), TARGET_RADIUS)

    # Show the score, average tap time, and average accuracy on the screen
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (WIDTH - 120, 10))

    # Update the display
    pygame.display.update()

    return retVal


def deinit() -> int:
    pygame.quit()
    return score


####################################################################################################
# MAIN                                                                                             #
####################################################################################################


def main() -> None:
    cv.cv_init(detect_fingers=True, detect_cds=False)
    init()

    loop_res: Loop_Result_t = Loop_Result_t.CONTINUE

    while loop_res == Loop_Result_t.CONTINUE:
        centroids: DetectedCentroids = cv.cv_loop()
        loop_res = loop(centroids)

    deinit()


if __name__ == "__main__":
    main()
