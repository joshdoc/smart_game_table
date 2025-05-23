####################################################################################################
# menu.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                              #
#                                                                                                  #
# This file runs the menu for our table.  It requires a dictionary of game names to game objects   #
# in its init() call.                                                                              #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

from math import ceil, sqrt
from typing import Optional

import pygame

from cv import DetectedCentroids
from sgt_types import Game, Game_t, Loop_Result_t

_selected_game: Optional[str] = None
_menu_items: list[tuple[pygame.Rect, str, pygame.Surface]] = []


####################################################################################################
# GLOBAL FUNCTIONS                                                                                 #
####################################################################################################


def init(args: tuple[dict[str, Game], Optional[str], Optional[int]]):
    global _font, _screen, _menu_items

    games: dict[str, Game] = args[0]
    prev_game_name: Optional[str] = args[1]
    prev_score: Optional[int] = args[2]

    pygame.init()
    _screen = pygame.display.set_mode((1400, 1050))
    pygame.display.set_caption("Select a Game")
    _font = pygame.font.SysFont("Arial", 32)

    # Show score summary if available
    if prev_game_name and prev_score is not None and prev_game_name in games:
        _screen.fill((0, 0, 0))

        game = games[prev_game_name]
        summary_lines = [
            f"{prev_game_name} Results",
            f"Score: {prev_score}",
            f"High Score: {game.high_score}",
        ]

        box_width, box_height = 600, 200
        box_x = (_screen.get_width() - box_width) // 2
        box_y = (_screen.get_height() - box_height) // 2
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)

        # Draw background box
        pygame.draw.rect(_screen, (255, 255, 255), box_rect, border_radius=20)

        # Render and draw text lines centered inside the box
        for i, line in enumerate(summary_lines):
            text_surface = _font.render(line, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(box_rect.centerx, box_rect.y + 40 + i * 50))
            _screen.blit(text_surface, text_rect)

        pygame.display.flip()
        pygame.time.wait(3000)
    else:
        # Avoid accidentally launching another game when escaping
        pygame.time.wait(1000)

    _screen.fill((0, 0, 0))

    _menu_items.clear()

    # Grid configuration
    num_games = len(games)
    columns = ceil(sqrt(num_games))  # square-ish layout
    rows = ceil(num_games / columns)
    box_margin = 20
    box_width = (1400 - (columns + 1) * box_margin) // columns
    box_height = (1050 - (rows + 1) * box_margin) // rows

    i = 0
    for row in range(rows):
        for col in range(columns):
            if i >= num_games:
                break

            game_name = list(games.keys())[i]
            game = games[game_name]
            display_text = f"{i + 1}. {game_name}"
            if game.game_type == Game_t.SCORED:
                display_text += f" (High Score: {game.high_score})"

            # Calculate box position
            x = box_margin + col * (box_width + box_margin)
            y = box_margin + row * (box_height + box_margin)

            # Draw box with rounded corners
            box_rect = pygame.Rect(x, y, box_width, box_height)
            pygame.draw.rect(_screen, (255, 255, 255), box_rect, border_radius=15)

            # Render text and center in box
            text_surf = _font.render(display_text, True, (0, 0, 0))
            text_rect = text_surf.get_rect(center=box_rect.center)

            _screen.blit(text_surf, text_rect)
            _menu_items.append((box_rect, game_name, text_surf))

            i += 1

    pygame.display.flip()


def loop(centroids: DetectedCentroids, _) -> Loop_Result_t:
    global _selected_game

    _screen.fill((0, 0, 0))

    retVal: Loop_Result_t = Loop_Result_t.CONTINUE

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _selected_game = None
            retVal = Loop_Result_t.EXIT

    for centroid in centroids.fingers:
        pygame.draw.circle(_screen, (0, 0, 255), (centroid.xpos, centroid.ypos), 15)
        for rect, game, _ in _menu_items:
            if rect.collidepoint((centroid.xpos, centroid.ypos)):
                _selected_game = game
                retVal = Loop_Result_t.EXIT

    for rect, _, surface in _menu_items:
        pygame.draw.rect(_screen, (255, 255, 255), rect, border_radius=15)
        text_rect = surface.get_rect(center=rect.center)
        _screen.blit(surface, text_rect)

    pygame.display.flip()

    return retVal


def deinit() -> Optional[str]:
    pygame.quit()
    return _selected_game


####################################################################################################
# MAIN                                                                                             #
####################################################################################################


if __name__ == "__main__":
    print("Error: This file is not intended to be run standalone.  Access through sgt.py.")
    exit()
