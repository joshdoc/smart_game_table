import pyautogui as pg

import cv

pg.FAILSAFE = False
scw, sch = pg.size()

# Scaling
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
        centroids = cv.cv_loop()
        if len(centroids.fingers):
            _move_mouse(centroids.fingers[0].xpos, centroids.fingers[0].ypos)


if __name__ == "__main__":
    main()
