import argparse
import csv
import math
import random
import time

import numpy as np
import pygame

from warp import cv_init, cv_loop

# Initialize Pygame
pygame.init()

# Game settings
WIDTH, HEIGHT = 1400, 1050
TARGET_SPAWN_RATE = 0.3  # New targets spawn every 1 second
TARGET_SPACING = 60
MAX_TARGETS = 10
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
TIME_TO_TAP = False

parser = argparse.ArgumentParser(description="Tap the Target Game")
parser.add_argument("duration", type=int, help="Duration of the game in seconds")
parser.add_argument("username", type=str, help="Username of the player")
parser.add_argument("radius", type=int, help="radius of the targets")
parser.add_argument("mouse", type=bool, help="mouse mode if true")

args = parser.parse_args()
GAME_DURATION = args.duration  # Duration from command line argument
USERNAME = args.username  # Username from command line argument
TARGET_RADIUS = args.radius
MOUSE_MODE = args.mouse


# Screen setup


# Game variables
targets = []  # List of targets
tap_times = []  # List to store tapping times
accuracies = []  # List to store accuracies
misses = 0
score = 0


# Function to calculate distance between two points
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate average tapping time
def avg_tap_time():
    if len(tap_times) > 0:
        return sum(tap_times) / len(tap_times)
    return 0


# Function to calculate median tapping time
def med_tap_time():
    if len(tap_times) > 0:
        return np.median(tap_times)
    return 0


# Function to calculate average accuracy
def avg_accuracy():
    if len(accuracies) > 0:
        return sum(accuracies) / len(accuracies)
    return 0


# Duration | User Name | Circle_Size | Num_Circles | Num_Misclisks | Avg_TBC | Med_TBC
def writeResults() -> None:
    with open("results.csv", mode="a", newline="") as file:
        out = csv.writer(file)
        out.writerow(
            [
                GAME_DURATION,
                USERNAME,
                TARGET_RADIUS,
                misses,
                avg_tap_time(),
                med_tap_time(),
            ]
        )


# Main game loop
if not MOUSE_MODE:
    cv_init()


def game():
    global misses, score
    screen = pygame.display.set_mode((1400, 1050))
    pygame.display.set_caption("Tap the Target Game")

    # Font for displaying text
    font = pygame.font.SysFont("Arial", 24)

    last_spawn_time = time.time()
    previous_tap_time = time.time()

    start_time = time.time()
    running = True
    completed = False
    while running:
        screen.fill(WHITE)

        # End the game if the time limit is reached
        elapsed_time = time.time() - start_time
        if elapsed_time >= GAME_DURATION:
            running = False
            completed = True

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if MOUSE_MODE and event.type == pygame.MOUSEBUTTONDOWN:
                miss = 1
                x, y = pygame.mouse.get_pos()

                for target in targets[:]:
                    target_x, target_y, creation_time = target
                    distance_from_center = distance(x, y, target_x, target_y)

                    # Check if click is within the target
                    if distance_from_center <= TARGET_RADIUS:
                        if TIME_TO_TAP:
                            tap_time = time.time() - creation_time
                        else:
                            tap_time = time.time() - previous_tap_time
                            previous_tap_time = time.time()
                        tap_times.append(tap_time)
                        accuracies.append(distance_from_center)
                        score += 1
                        targets.remove(target)  # Remove the target after successful hit
                        miss = 0
                misses += miss

        # ERRS
        # PG: 132 311
        # CV: 171 398

        if not MOUSE_MODE:
            # Centroid stuff
            centroids = cv_loop()
            print(centroids)
            for target in targets[:]:
                target_x, target_y, creation_time = target
                print(target_x, target_y)

                for centroid in centroids:
                    distance_from_center = distance(
                        centroid[0], centroid[1], target_x, target_y
                    )

                    # Check if click is within the target
                    if distance_from_center <= TARGET_RADIUS:
                        if TIME_TO_TAP:
                            tap_time = time.time() - creation_time
                        else:
                            tap_time = time.time() - previous_tap_time
                            previous_tap_time = time.time()
                        tap_times.append(tap_time)
                        accuracies.append(distance_from_center)
                        score += 1
                        if target in targets:
                            targets.remove(
                                target
                            )  # Remove the target after successful hit

        # Spawn new targets at the defined spawn rate
        if time.time() - last_spawn_time >= TARGET_SPAWN_RATE:
            last_spawn_time = time.time()
            if len(targets) < MAX_TARGETS:
                works = False
                while not works:
                    works = True
                    target_pos = (
                        random.randint(TARGET_RADIUS, WIDTH - TARGET_RADIUS),
                        random.randint(TARGET_RADIUS, HEIGHT - TARGET_RADIUS),
                    )
                    for target in targets:
                        if (
                            distance(target_pos[0], target_pos[1], target[0], target[1])
                            < TARGET_SPACING
                        ):
                            works = False
                            break
                    if works:
                        targets.append(
                            (target_pos[0], target_pos[1], time.time())
                        )  # Store target with spawn time

        # Draw all targets
        for target in targets:
            pygame.draw.circle(screen, RED, (target[0], target[1]), TARGET_RADIUS)

        # Show the score, average tap time, and average accuracy on the screen
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(score_text, (WIDTH - 120, 10))

        if TIME_TO_TAP:
            avg_tap_time_text = font.render(
                f"Avg Tap Time: {round(avg_tap_time(), 2)}s", True, (0, 0, 0)
            )
        else:
            avg_tap_time_text = font.render(
                f"Avg Time Between Taps: {round(avg_tap_time(), 2)}s", True, (0, 0, 0)
            )
        screen.blit(avg_tap_time_text, (10, 40))

        avg_accuracy_text = font.render(
            f"Avg Accuracy: {round(avg_accuracy(), 2)} px", True, (0, 0, 0)
        )
        screen.blit(avg_accuracy_text, (10, 70))

        # Update the display
        pygame.display.update()

    if completed:
        if not MOUSE_MODE:
            misses = input("num misses: ")
        writeResults()

    # Clean up and quit the game
    pygame.quit()


if __name__ == "__main__":
    game()
