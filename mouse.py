import pyautogui as pg

import cv

pg.FAILSAFE = False
scw, sch = pg.size()

# Mouse settings
X_SCALE: float = 3
Y_SCALE: float = 2.25

CFG_DEBUG: bool = False


def _move_mouse(xpos: int, ypos: int) -> None:
    pg.moveTo(xpos * X_SCALE, ypos * Y_SCALE)
    if CFG_DEBUG:
        print(f"Mouse moved to {xpos}, {ypos}")


def main():
    cv.cv_init()
    while True:
        position = cv.cv_loop()[0]
        _move_mouse(position[0], position[1])


if __name__ == "__main__":
    main()
