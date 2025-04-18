####################################################################################################
# mouse.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                             #
#                                                                                                  #
# This file uses the first detected finger centroid to move the mouse position, effectively making #
# the table act as a touchscreen.                                                                  #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

import pyautogui as pg

import cv
from sgt_types import Loop_Result_t

####################################################################################################
# PRIVATE FUNCTIONS                                                                                #
####################################################################################################


# Scaling unnecessary with 100% monitor scale in settings.
X_SCALE: float = 1
Y_SCALE: float = 1

# Print out mouse locations
CFG_DEBUG: bool = False


####################################################################################################
# GLOBALS                                                                                          #
####################################################################################################


pg.FAILSAFE = False
scw, sch = pg.size()


####################################################################################################
# PRIVATE FUNCTIONS                                                                                #
####################################################################################################


def _move_mouse(xpos: int, ypos: int) -> None:
    pg.moveTo(xpos * X_SCALE, ypos * Y_SCALE)
    pg.click()
    if CFG_DEBUG:
        print(f"Mouse moved to {xpos}, {ypos}")


####################################################################################################
# GAME API                                                                                         #
####################################################################################################


def loop(centroids: cv.DetectedCentroids, _=None) -> Loop_Result_t:
    retVal: Loop_Result_t = Loop_Result_t.CONTINUE

    if centroids.escape:
        retVal = Loop_Result_t.EXIT

    if len(centroids.fingers):
        _move_mouse(centroids.fingers[0].xpos, centroids.fingers[0].ypos)

    return retVal


####################################################################################################
# MAIN                                                                                             #
####################################################################################################


def main():
    cv.cv_init()
    while True:
        centroids = cv.cv_loop()
        loop(centroids)


if __name__ == "__main__":
    main()
