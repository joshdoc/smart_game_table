####################################################################################################
# cv.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                                #
#                                                                                                  #
# This file is used to detect finger presses and tangible object locations on our EECS 498/598     #
# Engineering Interactive Systems project.                                                         #
#                                                                                                  #
# To use: import this file, then call cv_init(), followed by cv_loop().  cv_loop() should be       #
# be called anytime you want to get new centroid detections, but should not be called faster than  #
# the camera refresh rate (in our case 30 Hz)                                                      #
#                                                                                                  #
####################################################################################################

####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

from sgt_types import Centroid, DetectedCentroids

####################################################################################################
# TYPES                                                                                            #
####################################################################################################

@dataclass
class HoverParams:
    inner_threshold:int
    outer_threshold: int
    offset: int
    grid_size: int

    hover_grid: cv2.typing.MatLike


@dataclass
class DetectionParameters:
    inner_threshold: int
    outer_threshold: int
    min_area: int
    max_area: int
    detect: bool


####################################################################################################
# CONSTANTS                                                                                        #
####################################################################################################


# Run `ls /dev | grep video` to see which idx to use for the camera (or guess on windows)
CAM_IDX: int = 0

# USAGE: frame[CROP]         Y1:Y2      , X1:X2
CROP: tuple[slice, slice] = (slice(0, 1), slice(0, 1))

CROP_SCALE: float = 0.975  # scale the cropped image
CROP_MIN_THRESH: int = 10  # threshold for detecting the edges

# define screen size for warping the image
WIDTH = 1400
HEIGHT = 1050
TARGET = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]

# offsets for correcting warp errors
CUT_LOW = 15
CUT_RIGHT = -10
CUT_LEFT = -10
CUT_TOP = -10 - 15
TOP_LEFT_CORRECTION_FACTOR_X: int = 10
TOP_LEFT_CORRECTION_FACTOR_Y: int = 10

# Number of tangibles expected
## 2 for air hockey
N_TANGIBLES = 2

# Parameters for centroid detection
HULL_MIN_SOLIDITY: float = 0.8

FINGER_INNER_THRESHOLD: int = 25  # 47
FINGER_OUTER_THRESHOLD: int = 31  # 69

FINGER_MIN_AREA: int = 100
FINGER_MAX_AREA: int = 1500

HOVER_INNER_THRESHOLD: int = 15
HOVER_OUTER_THRESHOLD: int = 15

CD_INNER_THRESHOLD: int = 24  # 35
CD_OUTER_THRESHOLD: int = 29  # 42
CD_MIN_AREA: int = 26 * 1000  # 40000
CD_MAX_AREA: int = 40 * 1000  # 60000

FINGER_PARAMS = DetectionParameters(
    FINGER_INNER_THRESHOLD, FINGER_OUTER_THRESHOLD, FINGER_MIN_AREA, FINGER_MAX_AREA, False
)
CD_PARAMS = DetectionParameters(
    CD_INNER_THRESHOLD, CD_OUTER_THRESHOLD, CD_MIN_AREA, CD_MAX_AREA, False
)

# Text options
TEXT_OPTS = [cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2]


####################################################################################################
# CONFIGURATION                                                                                    #
####################################################################################################


# Show the initial cropped image (2 seconds)
CFG_SHOW_INITIAL_CROP: bool = False
# Show the background subtracted image (each call to cv_loop)
CFG_SHOW_BG_SUBTRACT: bool = False
# Show the cropped background (2 seconds)
CFG_SHOW_INITIAL_BG: bool = False
# If running from another application, show frame if this is true (each call to cv_loop)
CFG_SHOW_FRAME: bool = False
# Show FPS in frame (each call to cv_loop)
CFG_SHOW_FPS: bool = True
# Use the trackbars for settings
CFG_USE_TRACKBARS: bool = False
# draw the contours on the frame
CFG_SHOW_CENTROIDS: bool = False
# Show the thresholded frame
CFG_SHOW_THRESH: bool = False
CFG_SHOW_HOVER_THRESH: bool = True
# Draw the detected hulls
CFG_SHOW_HULL: bool = False


####################################################################################################
# GLOBALS                                                                                          #
####################################################################################################


corners: np.ndarray = np.zeros(0)
standalone: bool = False
capture: cv2.VideoCapture = cv2.VideoCapture()
bg: np.ndarray = np.zeros(0)
current_margin: int = 15

frame_count: int = 0
fps: float = 0
start_time: float = time.time()
lastPos: list[tuple[int, int]] = []
use_hover: bool = False

hoversettings: HoverParams = None


####################################################################################################
# CALLBACKS                                                                                        #
####################################################################################################


def _update_contours(margin: int) -> None:
    global current_margin
    current_margin = margin


####################################################################################################
# LOCAL FUNCTIONS                                                                                  #
####################################################################################################


### Define a function to reorder rectangle corners
def _reorder_corners(corners: np.ndarray) -> np.ndarray:
    threshold = 200
    new_corners = copy.deepcopy(corners)
    for cor in corners:  # Strange ordering to line up with table
        if cor[0][0] < threshold and cor[0][1] < threshold:
            new_corners[1] = cor
        elif cor[0][0] > threshold and cor[0][1] < threshold:
            new_corners[0] = cor
        elif cor[0][0] > threshold and cor[0][1] > threshold:
            new_corners[3] = cor
        else:
            new_corners[2] = cor
    # improve edges
    new_corners[2][0][1] -= CUT_LOW
    new_corners[3][0][1] -= CUT_LOW
    new_corners[0][0][0] -= CUT_RIGHT
    new_corners[3][0][0] -= CUT_RIGHT
    new_corners[1][0][0] -= CUT_LEFT
    new_corners[2][0][0] -= CUT_LEFT
    new_corners[2][0][1] -= CUT_TOP
    new_corners[3][0][1] -= CUT_TOP
    return new_corners

### Get the corners of the table for cropping the image
def _crop_bg(frame: np.ndarray) -> None:
    global corners

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, CROP_MIN_THRESH, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # largest contour is the table
    if len(contours):
        table_outline = max(contours, key=cv2.contourArea)

        peri = cv2.arcLength(table_outline, True)
        corners = cv2.approxPolyDP(table_outline, 0.04 * peri, True)
        cv2.polylines(frame, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)
        corners = _reorder_corners(corners)

    if CFG_SHOW_INITIAL_BG:

        cv2.imshow("frame", frame)
        cv2.waitKey(2000)


### Correct camera perspective to match table top
def _warp_image(image: np.ndarray) -> np.ndarray:
    corners_np = np.array(corners, dtype=np.float32)
    target_np = np.array(TARGET, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(corners_np, target_np)
    out = cv2.warpPerspective(image, mat, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
    if CFG_SHOW_INITIAL_CROP:
        cv2.namedWindow("out", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("out", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("out", out)
        cv2.waitKey(2000)
    return out


### Capture initial background for background subtraction
def _capture_bg(capture: cv2.VideoCapture) -> np.ndarray:
    global bg

    print("Press 'b' to capture the background frame.")
    capturing: bool = True
    while capturing:
        ret, frame = capture.read()
        frame = _warp_image(frame)
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            exit()

        cv2.imshow("Live Feed - Press 'b' to capture background", frame)

        if cv2.waitKey(1) == ord("b"):
            bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Background captured.")
            capturing = False

    cv2.destroyWindow("Live Feed - Press 'b' to capture background")
    return bg


def _capture_init(default_values: bool = True) -> None:
    global capture

    capture = cv2.VideoCapture(CAM_IDX)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not default_values:
        capture.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        capture.set(cv2.CAP_PROP_CONTRAST, 10)
        capture.set(cv2.CAP_PROP_SATURATION, 0)
        capture.set(cv2.CAP_PROP_SHARPNESS, 0)
        capture.set(cv2.CAP_PROP_BACKLIGHT, 0)
        capture.set(cv2.CAP_PROP_ZOOM, 0)
        capture.set(cv2.CAP_PROP_EXPOSURE, -8)
        capture.set(cv2.CAP_PROP_PAN, 0)
        capture.set(cv2.CAP_PROP_TILT, 0)
        time.sleep(3)

    if not capture.isOpened():
        print("Error: Could not open the camera.")
        exit()


# initialize trackbar window
def _trackbar_init() -> None:
    ret, frame = capture.read()
    if not ret:
        print("Error: Could not read frame.")
        exit()

    height, width = frame.shape[:2]

    cv2.namedWindow("Controls", cv2.WND_PROP_FULLSCREEN)
    cv2.createTrackbar("Margin", "Controls", 10, min(height, width) // 2, _update_contours)

    control_image = cv2.imread("debug/control.png")
    cv2.imshow("Controls", control_image)

    cv2.createTrackbar("ThreshFingerI", "Controls", 0, 255, lambda _: None)
    cv2.setTrackbarPos("ThreshFingerI", "Controls", 47)  # inner
    cv2.createTrackbar("ThreshFingerO", "Controls", 0, 255, lambda _: None)
    cv2.setTrackbarPos("ThreshFingerO", "Controls", 69)  # outer

    cv2.createTrackbar("ThreshIcd", "Controls", 0, 255, lambda _: None)
    cv2.setTrackbarPos("ThreshIcd", "Controls", 42)  # inner
    cv2.createTrackbar("ThreshOcd", "Controls", 0, 255, lambda _: None)
    cv2.setTrackbarPos("ThreshOcd", "Controls", 42)  # outer


def _threshold(diff: np.ndarray, inner_thresh, outer_thresh) -> np.ndarray:
    topLX = 313
    topLY = 198
    height, width = diff.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (topLX, topLY), (width - topLX, height - topLY), 255, -1)
    inner = cv2.bitwise_and(diff, mask)
    outer = cv2.bitwise_and(diff, cv2.bitwise_not(mask))

    # Different thresholds for the different sections
    if CFG_USE_TRACKBARS:
        inner_thresh = cv2.getTrackbarPos("ThreshFingerI", "Controls")
        outer_thresh = cv2.getTrackbarPos("ThreshFingerO", "Controls")

    _, threshI = cv2.threshold(inner, inner_thresh, 255, cv2.THRESH_BINARY)
    _, threshO = cv2.threshold(outer, outer_thresh, 255, cv2.THRESH_BINARY)

    thresh = cv2.add(threshI, threshO)
    return thresh


def _top_left_corner_correction(x: int, y: int) -> tuple[int, int]:
    if x < 200 and y < 200:
        x -= 10
        y -= 10
    elif x < 800 and y < 200:
        x -= 10
    return x, y


# escape sequences will be triggered when there is a centroid within two corners of the table
def _detect_escape(centroids: list[Centroid]) -> bool:
    # Define the coordinates of the four corners of the table
    top_left = (0, 0)
    top_right = (WIDTH, 0)
    bottom_left = (0, HEIGHT)
    bottom_right = (WIDTH, HEIGHT)

    # Define escape areas around the corners with the given margin
    margin = 100
    escape_top_left = (top_left[0] + margin, top_left[1] + margin)
    escape_top_right = (top_right[0] - margin, top_right[1] + margin)
    escape_bottom_left = (bottom_left[0] + margin, bottom_left[1] - margin)
    escape_bottom_right = (bottom_right[0] - margin, bottom_right[1] - margin)

    esc_corners: set[str] = set()

    for centroid in centroids:
        if centroid.xpos <= escape_top_left[0] and centroid.ypos <= escape_top_left[1]:
            esc_corners.add("top_left")
        elif centroid.xpos >= escape_top_right[0] and centroid.ypos <= escape_top_right[1]:
            esc_corners.add("top_right")
        elif centroid.xpos <= escape_bottom_left[0] and centroid.ypos >= escape_bottom_left[1]:
            esc_corners.add("bottom_left")
        elif centroid.xpos >= escape_bottom_right[0] and centroid.ypos >= escape_bottom_right[1]:
            esc_corners.add("bottom_right")

    return len(esc_corners) >= 2


def _detect_centroids(contours: Sequence[MatLike], min_area: int, max_area: int) -> list[Centroid]:
    centroids: list[Centroid] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        if solidity > HULL_MIN_SOLIDITY and min_area < hull_area and hull_area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cX, cY = _top_left_corner_correction(cX, cY)

                centroids.append(Centroid(cX, cY, hull))

    return centroids


def _run_detection(img: np.ndarray, params: DetectionParameters) -> list[Centroid]:
    if not params.detect:
        return []

    thresh = _threshold(img, params.inner_threshold, params.outer_threshold)

    # Use morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop over the contours to detect and draw centroids
    centroids = _detect_centroids(contours, params.min_area, params.max_area)

    if CFG_SHOW_THRESH:
        cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Thresh", cv2.WND_PROP_FULLSCREEN, 1)
        cv2.imshow("Thresh", thresh)

    return centroids


def _distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _idCD(cds: list[Centroid]) -> list[Centroid]:
    global lastPos
    newCDlist: list[Centroid] = [Centroid(0, 0, np.zeros(0)) for _ in range(N_TANGIBLES)]
    for cd in cds:
        correctPosition = 9  # placeholder max values
        maxDist = 99999999  # placeholder max values
        if len(lastPos) < N_TANGIBLES:
            # time.sleep(2)
            lastPos.append((cd.xpos, cd.ypos))
        for i in range(len(lastPos)):
            d = _distance(lastPos[i][0], lastPos[i][1], cd.xpos, cd.ypos)
            if d < maxDist:
                correctPosition = i
                maxDist = d
        # print("CorrectPos", correctPosition)
        newCDlist[correctPosition] = cd
        lastPos[correctPosition] = (cd.xpos, cd.ypos)
    return newCDlist


####################################################################################################
# GLOBAL FUNCTIONS                                                                                 #
####################################################################################################


def toggle_hover(offset, inner_threshold=15, outer_threshold=15, scale_size=20):
    global use_hover, hoversettings

    if hoversettings is None:
        hoversettings = HoverParams(inner_threshold,outer_threshold,offset,scale_size,np.ndarray(0))
       
    use_hover = True

def detect_hover(img) -> cv2.typing.MatLike:
    global hover
    cropped_image = img[hoversettings.offset:HEIGHT-(2*hoversettings.offset), 0:WIDTH]
    thresh = _threshold(cropped_image, hoversettings.inner_threshold, hoversettings.outer_threshold)

    # Use morphological operations to remove small noise
    #kernel = np.ones((3, 3), np.uint8)

    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    
    img_temp = cv2.resize(thresh, (hoversettings.grid_size, hoversettings.grid_size), interpolation=cv2.INTER_AREA)     

    if CFG_SHOW_HOVER_THRESH:
        cv2.namedWindow("Hover Thresh", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Hover Thresh", cv2.WND_PROP_FULLSCREEN, 1)
        cv2.imshow("Hover Thresh", img_temp)

    return img_temp

### Initialize the program
def cv_init(detect_fingers: bool = True, detect_cds: bool = True) -> None:
    _capture_init()

    FINGER_PARAMS.detect = detect_fingers
    CD_PARAMS.detect = detect_cds

    # Automatically crop image
    ret, frame = capture.read()

    if not ret:
        print("Error: Could not read frame.")
        exit()

    _crop_bg(frame)

    # Capture the background scene for subtraction
    _capture_bg(capture)

    if CFG_USE_TRACKBARS:
        _trackbar_init()


### Detect centroids (finger presses) and return list
def cv_loop() -> DetectedCentroids:
    global fps, frame_count, start_time
    global hoversettings, use_hover

    ret, frame = capture.read()

    retVal = DetectedCentroids([], [], False)

    if not ret:
        print("Error: Could not read frame.")
        exit()

    frame = _warp_image(frame)

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the background and current frame
    diff = cv2.absdiff(bg, gray_frame)

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (current_margin, current_margin), (WIDTH - current_margin, HEIGHT - current_margin), 255, -1)
    #cv2.rectangle(mask, (topLX, topLY), (width - topLX, height - topLY), 255, -1)

    diff = cv2.bitwise_and(diff, (mask))

    retVal.fingers = _run_detection(diff, FINGER_PARAMS)
    retVal.cds = _run_detection(diff, CD_PARAMS)
    if use_hover:
        print("running hover")
        hoversettings.hover_grid = detect_hover(diff)

    # N_TANGIBLES should be configurable in init in the future
    if len(retVal.cds) == N_TANGIBLES:
        retVal.cds = _idCD(retVal.cds)

    if len(retVal.fingers):
        retVal.escape = _detect_escape(retVal.fingers)

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time

    # update FPS each second
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        # reset counters
        start_time = current_time
        frame_count = 0

    # Display the FPS on the frame
    if CFG_SHOW_FPS:
        fps_display = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_display, (10, 50), *TEXT_OPTS)

    if CFG_SHOW_CENTROIDS:
        for centroid in retVal.fingers:
            cv2.drawContours(frame, [centroid.contour_hull], 0, (255, 255, 0), 2)
            cv2.circle(frame, (centroid.xpos, centroid.ypos), 5, (0, 255, 255), -1)
        centroidNum = 0
        for centroid in retVal.cds:
            cv2.circle(frame, (centroid.xpos, centroid.ypos), 5, (0, 0, 255), -1)
            if centroidNum == 0:
                cv2.drawContours(frame, [centroid.contour_hull], 0, (255, 0, 255), 2)
            elif centroidNum == 1:
                cv2.drawContours(frame, [centroid.contour_hull], 0, (0, 255, 255), 2)
            else:
                cv2.drawContours(frame, [centroid.contour_hull], 0, (255, 0, 0), 2)
            centroidNum += 1

        if CFG_SHOW_HULL:
            for centroid in retVal.cds:
                cv2.drawContours(frame, [centroid.contour_hull], 0, (255, 0, 255), 2)
                cv2.circle(frame, (centroid.xpos, centroid.ypos), 5, (0, 255, 255), -1)
                txt = "Hull Area: " + str(cv2.contourArea(centroid.contour_hull))
                cv2.putText(frame, txt, (centroid.xpos, centroid.ypos), *TEXT_OPTS)

    # Display original frame with detected centroids and the threshold image
    if standalone or CFG_SHOW_FRAME:
        cv2.namedWindow("Detected Centroids", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detected Centroids", cv2.WND_PROP_FULLSCREEN, 1)
        cv2.imshow("Detected Centroids", frame)

    if CFG_SHOW_BG_SUBTRACT:
        cv2.imshow("Background Subtraction", diff)

    # Exit the loop when 'q' is pressed if running standalone
    if standalone and cv2.waitKey(1) == ord("q"):
        capture.release()
        cv2.destroyAllWindows()
        exit()

    return retVal


####################################################################################################
# MAIN                                                                                             #
####################################################################################################


def main() -> None:
    global standalone
    standalone = True

    cv_init(detect_fingers=True, detect_cds=True)

    #toggle_hover(139, 15, 15, 20)

    while True:
        cv_loop()


if __name__ == "__main__":
    main()
