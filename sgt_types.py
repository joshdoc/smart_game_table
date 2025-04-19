####################################################################################################
# sgt_types.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                         #
#                                                                                                  #
# This file contains all of the shared types needed for the smart game table games.                #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

import importlib
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Callable, Optional

import numpy as np

####################################################################################################
# TYPES                                                                                            #
####################################################################################################


class Loop_Result_t(Enum):
    CONTINUE = 1
    EXIT = 2


@dataclass
class Centroid:
    xpos: int
    ypos: int
    contour_hull: np.ndarray


@dataclass
class DetectedCentroids:
    fingers: list[Centroid]
    cds: list[Centroid]
    escape: bool


class Game_t(Enum):
    MENU = 1
    SCORED = 2
    UNSCORED = 3


@dataclass
class Game:
    name: str
    game_type: Game_t
    high_score: int = field(init=False)
    init: Callable[[Optional[Any]], None] = field(init=False)
    loop: Callable[[DetectedCentroids, float], Loop_Result_t] = field(init=False)
    # return type depends on game type
    deinit: Callable[[], Optional[int | str]] = field(init=False)
    _module: ModuleType = field(init=False)

    def __post_init__(self):
        # import the module
        self._module = importlib.import_module(self.name)

        try:
            self.init = getattr(self._module, "init", lambda _: None)
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
