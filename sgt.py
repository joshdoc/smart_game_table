####################################################################################################
# sgt.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                               #
#                                                                                                  #
# This file is responsible for calling our games and user tests from a GUI controlled via our      #
# touchscreen.                                                                                     #
#                                                                                                  #
# Game API:                                                                                        #
# Required Functions:                                                                              #
#                                                                                                  #
# loop(cv.DetectedCentroids, float) -> bool                                                        #
# loop is called once per frame to run the game, it is passed in dt which is the elapsed time      #
# since the last frame and centroids which is the DetectedCentroids object returned by cv_loop.    #
# It must return a boolean specifying if the game should exit (True) or continue running (False).  #
# Typically, this can be achieved by using the DetectedCentroids.escape field which is True in the #
# event that there are two finger presses in the corners of the table.                             #
#                                                                                                  #
# Optional Functions:                                                                              #
# init() -> None                                                                                   #
# init is called once before the game loop.  Use this function to initialize globals, etc.         #
#                                                                                                  #
# deinit() -> int|None                                                                             #
# deinit is called once after the game loop exits.  If the game is scored, return the score from   #
# this function to have this file maintain a running scoreboard (if Game.score = True).            #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

import importlib
import time
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Callable, Optional

import cv
from sgt_types import Loop_Result_t

####################################################################################################
# TYPES                                                                                            #
####################################################################################################


class Game_t(Enum):
    MENU = 1
    SCORED = 2
    UNSCORED = 3


@dataclass
class Game:
    name: str
    game_type: Game_t
    high_score: int = field(init=False)
    init: Callable[[], None] = field(init=False)
    loop: Callable[[cv.DetectedCentroids, float], Loop_Result_t] = field(init=False)
    # return type depends on game type
    deinit: Callable[[], Optional[int | str]] = field(init=False)
    _module: ModuleType = field(init=False)

    def __post_init__(self):
        # import the module
        self._module = importlib.import_module(self.name)

        try:
            self.init = getattr(self._module, "init", lambda: None)
            self.loop = getattr(self._module, "loop")
            self.deinit = getattr(self._module, "deinit", lambda: None)
        except AttributeError as e:
            raise ImportError("Module '{}' must define 'loop'".format(self.name)) from e

        # High score tracking
        self.high_score = 0
        try:
            with open(".{}".format(self.name), "r") as f:
                content = f.read().strip()
                self.high_score = int(content)
        except FileNotFoundError:
            print("Error: File '.{}' not found.".format(self.name))
        except ValueError:
            print("Error: File '.{}' does not contain a valid integer.".format(self.name))

    def update_high_score(self, score: int) -> None:
        self.high_score = score
        try:
            with open(f".{self.name}", "w") as f:
                f.write(str(score))
        except Exception as e:
            print(f"Error writing high score for '{self.name}': {e}")


####################################################################################################
# CONSTANTS                                                                                        #
####################################################################################################


MENU = Game(name="menu", game_type=Game_t.MENU)
UNDERTABLE = Game(name="undertable", game_type=Game_t.SCORED)
HOCKEY = Game(name="hockey", game_type=Game_t.UNSCORED)
USER_TEST = Game(name="user_test", game_type=Game_t.SCORED)
MOUSE = Game(name="mouse", game_type=Game_t.UNSCORED)

GAMES = {game.name: game for game in [UNDERTABLE, HOCKEY, USER_TEST, MOUSE]}


####################################################################################################
# LOCAL FUNCTIONS                                                                                  #
####################################################################################################


def _game_return_type_error(game: Game, ret_type: type | None) -> None:
    correct_type: type | None = None
    if game.game_type == Game_t.MENU:
        correct_type = str
    if game.game_type == Game_t.SCORED:
        correct_type = int
    raise ValueError("Module '{}' must return {}, not {}".format(game.name, correct_type, ret_type))


####################################################################################################
# MAIN                                                                                             #
####################################################################################################


def main() -> None:
    cur_game: Game = MENU
    dt: float = 0
    prev_time = time.time()
    loop_result: Loop_Result_t = Loop_Result_t.CONTINUE
    game_result: Optional[int | str] = None
    invalid_game_result: bool = False

    cv.cv_init(detect_fingers=True, detect_cds=True)

    while True:
        centroids = cv.cv_loop()

        dt = time.time() - prev_time
        loop_result = cur_game.loop(centroids, dt)
        prev_time = time.time()

        if loop_result == Loop_Result_t.EXIT:
            game_result = cur_game.deinit()

            if cur_game.game_type == Game_t.MENU:
                if type(game_result) is str:
                    cur_game = GAMES[game_result]
                else:
                    invalid_game_result = True

            if cur_game.game_type == Game_t.SCORED:
                if type(game_result) is int and game_result > cur_game.high_score:
                    cur_game.update_high_score(game_result)
                else:
                    invalid_game_result = True

            if invalid_game_result:
                _game_return_type_error(cur_game, type(game_result))


if __name__ == "__main__":
    main()
