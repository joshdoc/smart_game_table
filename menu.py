from typing import Optional

import pygame

from cv import DetectedCentroids
from sgt_types import Game, Game_t, Loop_Result_t

_selected_game: Optional[str] = None
_font = None
_screen = None
_menu_items = []  # List of (rect, game_name) pairs


def init(games: dict[str,Game]):
    global _initialized, _font, _screen, _menu_items

    pygame.init()
    _screen = pygame.display.set_mode((1400, 1050))

    pygame.display.set_caption("Select a Game")
    _font = pygame.font.SysFont("Arial", 32)
    _screen.fill((0, 0, 0))

    _menu_items.clear()
    padding = 20
    item_height = 50
    for i, game in enumerate(games):
        text = f"{i + 1}. {game}"
        if games[game].game_type == Game_t.SCORED:
            text += f" (High Score: {games[game].high_score})"

        surface = _font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(topleft=(100, 100 + i * (item_height + padding)))
        _screen.blit(surface, rect)
        _menu_items.append((rect, game, surface))

    pygame.display.flip()
    _initialized = True


def loop(centroids: DetectedCentroids, _) -> Loop_Result_t:
    global _selected_game

    _screen.fill((0,0,0))

    retVal: Loop_Result_t = Loop_Result_t.CONTINUE

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _selected_game = None
            retVal = Loop_Result_t.EXIT

    for centroid in centroids.fingers:
        pygame.draw.circle(_screen, (0,0,255), (centroid.xpos, centroid.ypos), 15)
        for rect, game, _ in _menu_items:
            if rect.collidepoint((centroid.xpos, centroid.ypos)):
                _selected_game = game
                retVal = Loop_Result_t.EXIT

    for rect, game, surface in _menu_items:
        _screen.blit(surface, rect)


    pygame.display.flip()

    return retVal


def deinit() -> Optional[str]:
    pygame.quit()
    return _selected_game
