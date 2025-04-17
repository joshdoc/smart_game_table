from typing import Optional

import pygame

from cv import DetectedCentroids

# Mocked game list and high scores (normally provided by `sgt.py`)
_games = ["undertable", "hockey", "user_test", "mouse"]
_high_scores = {
    "undertable": 42,
    "user_test": 87,
}

_selected_game: Optional[str] = None
_initialized: bool = False
_font = None
_screen = None
_menu_items = []  # List of (rect, game_name) pairs


def init():
    global _initialized, _font, _screen, _menu_items
    if _initialized:
        return

    pygame.init()
    _screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Select a Game")
    _font = pygame.font.SysFont("Arial", 32)
    _screen.fill((0, 0, 0))

    _menu_items.clear()
    padding = 20
    item_height = 50
    for i, game in enumerate(_games):
        text = f"{i+1}. {game}"
        if game in _high_scores:
            text += f" (High Score: {_high_scores[game]})"

        surface = _font.render(text, True, (255, 255, 255))
        rect = surface.get_rect(topleft=(100, 100 + i * (item_height + padding)))
        _screen.blit(surface, rect)
        _menu_items.append((rect, game))

    pygame.display.flip()
    _initialized = True


def loop(centroids: DetectedCentroids, dt: float) -> bool:
    global _selected_game

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _selected_game = None
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for rect, game in _menu_items:
                if rect.collidepoint(pos):
                    _selected_game = game
                    return True

    return centroids.escape  # fallback ESC gesture


def deinit() -> str:
    pygame.quit()
    return _selected_game or _games[0]
