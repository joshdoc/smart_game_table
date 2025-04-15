import pyautogui as pg

import cv

pg.FAILSAFE = False
scw, sch = pg.size()

# Mouse settings
X_SCALE: float = 1
Y_SCALE: float = 1

CFG_DEBUG: bool = False


def _move_mouse(xpos: int, ypos: int) -> None:
    pg.moveTo(xpos * X_SCALE, ypos * Y_SCALE)
    pg.click()
    if CFG_DEBUG:
        print(f"Mouse moved to {xpos}, {ypos}")


def main():
    cv.cv_init()
    while True:
        position = cv.cv_loop()
        if len(position):
            _move_mouse(position[0][0], position[0][1])


if __name__ == "__main__":
    main()
