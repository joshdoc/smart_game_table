from typing import Optional

import pygame

from cv import DetectedCentroids
from sgt_types import Game, Game_t, Loop_Result_t

_selected_game: Optional[str] = None
_font = None
_screen = None
_menu_items = []  # List of (rect, game_name) pairs


def init(games: list[Game]):
    global _initialized, _font, _screen, _menu_items

    pygame.init()
    _screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Select a Game")
    _font = pygame.font.SysFont("Arial", 32)
    _screen.fill((0, 0, 0))

    _menu_items.clear()
    padding = 20
    item_height = 50
    for i, game in enumerate(games):
        text = f"{i + 1}. {game.name}"
        if game.game_type == Game_t.SCORED:
            text += f" (High Score: {game.high_score})"

        surface = _font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(topleft=(100, 100 + i * (item_height + padding)))
        _screen.blit(surface, rect)
        _menu_items.append((rect, game.name))

    pygame.display.flip()
    _initialized = True


def loop(centroids: DetectedCentroids, _) -> Loop_Result_t:
    global _selected_game

    retVal: Loop_Result_t = Loop_Result_t.CONTINUE

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _selected_game = None
            retVal = Loop_Result_t.EXIT

    if centroids.escape:
        _selected_game = None
        retVal = Loop_Result_t.EXIT

    for centroid in centroids.fingers:
        for rect, game in _menu_items:
            if rect.collidepoint((centroid.xpos, centroid.ypos)):
                _selected_game = game.name
                retVal = Loop_Result_t.EXIT

    return retVal


def deinit() -> Optional[str]:
    pygame.quit()
    return _selected_game
