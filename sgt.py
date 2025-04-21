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
# init(_) -> None                                                                                  #
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

import time
from typing import Optional
import os

import cv
from sgt_types import Game, Game_t, Loop_Result_t

####################################################################################################
# CONSTANTS                                                                                        #
####################################################################################################


MENU = Game(name="menu", game_type=Game_t.MENU)
UNDERTABLE = Game(name="undertable_touch", game_type=Game_t.SCORED)
HOCKEY = Game(name="hockey", game_type=Game_t.UNSCORED)
DOTS = Game(name="dots", game_type=Game_t.SCORED)
MOUSE = Game(name="mouse", game_type=Game_t.UNSCORED)
MACRODAT = Game(name="macrodata", game_type=Game_t.UNSCORED)
DEBUG = Game(name="debug", game_type=Game_t.UNSCORED)

GAMES = {game.name: game for game in [DOTS, HOCKEY, UNDERTABLE, MOUSE, MACRODAT]}


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

    if (cur_game.name == "debug"):
        os.system('cv.py')
        cur_game = MENU


    cv.cv_init(detect_fingers=True, detect_cds=True)
    cur_game.init((GAMES, None, None))

    while True:
        prev_game_name: Optional[str] = None
        prev_game_score: Optional[int] = None
        centroids = cv.cv_loop()

        dt = time.time() - prev_time
        prev_time = time.time()
        loop_result = cur_game.loop(centroids, dt)

        if loop_result == Loop_Result_t.EXIT:
            game_result = cur_game.deinit()

            if cur_game.game_type == Game_t.MENU:
                if type(game_result) is str:
                    cur_game = GAMES[game_result]
                else:
                    invalid_game_result = True

            elif cur_game.game_type == Game_t.SCORED:
                if type(game_result) is int:
                    if game_result > cur_game.high_score:
                        cur_game.update_high_score(game_result)
                    prev_game_name = cur_game.name
                    prev_game_score = game_result
                    cur_game = MENU
                elif type(game_result) is not int:
                    invalid_game_result = True

            else:
                cur_game = MENU

            if invalid_game_result:
                _game_return_type_error(cur_game, type(game_result))

            cur_game.init((GAMES, prev_game_name, prev_game_score) if cur_game == MENU else None)


if __name__ == "__main__":
    main()
