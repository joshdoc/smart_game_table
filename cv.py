####################################################################################################
# CV.py | Authors: mcprisk, dcalco, joshdoc, neelnv                                                #
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
# Imports                                                                                          #
####################################################################################################

import copy
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

####################################################################################################
# Types                                                                                            #
####################################################################################################


@dataclass
class Centroid:
    xpos: int
    ypos: int


@dataclass
class DetectedCentroids:
    fingers: list[Centroid]
    cds: list[Centroid]


@dataclass
class DetectionParameters:
    inner_threshold: int
    outer_threshold: int
    min_area: int
    max_area: int
    detect: bool


####################################################################################################
# Constants                                                                                        #
####################################################################################################


# Run `ls /dev | grep video` to see which idx to use for the camera (or guess on windows)
CAM_IDX: int = 0

# USAGE: frame[CROP]         Y1:Y2      , X1:X2
CROP: tuple[slice, slice] = (slice(0, 1), slice(0, 1))

CROP_SCALE: float = 0.975  # scale the cropped image
CROP_MIN_THRESH: int = 30  # threshold for detecting the edges

# define screen size for warping the image
WIDTH = 1400
HEIGHT = 1050
TARGET = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]

# offsets for correcting warp errors
CUT_LOW = 15
CUT_RIGHT = 5
CUT_LEFT = 5
CUT_TOP = -10

# Parameters for centroid detection
HULL_MIN_SOLIDITY: float = 0.8

FINGER_INNER_THRESHOLD: int = 47
FINGER_OUTER_THRESHOLD: int = 69
FINGER_MIN_AREA: int = 100
FINGER_MAX_AREA: int = 1500

CD_INNER_THRESHOLD: int = 42
CD_OUTER_THRESHOLD: int = 42
CD_MIN_AREA: int = 40000
CD_MAX_AREA: int = 55000

FINGER_PARAMS = DetectionParameters(
    FINGER_INNER_THRESHOLD, FINGER_OUTER_THRESHOLD, FINGER_MIN_AREA, FINGER_MAX_AREA, False
)
CD_PARAMS = DetectionParameters(
    CD_INNER_THRESHOLD, CD_OUTER_THRESHOLD, CD_MIN_AREA, CD_MAX_AREA, False
)

# Text options
TEXT_OPTS = [cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2]


####################################################################################################
# Configuration                                                                                    #
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


####################################################################################################
# Globals                                                                                          #
####################################################################################################


corners: np.ndarray = np.zeros(0)
standalone: bool = False
capture: cv2.VideoCapture = cv2.VideoCapture()
bg: np.ndarray = np.zeros(0)
current_margin: int = 0

frame_count: int = 0
fps: float = 0
start_time = time.time()


####################################################################################################
# Callbacks                                                                                        #
####################################################################################################


def _update_contours(margin):
    global current_margin
    current_margin = margin


def _nothing(_):
    pass


####################################################################################################
# Private Functions                                                                                #
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

    cv2.createTrackbar("ThreshI", "Controls", 0, 255, _nothing)
    cv2.setTrackbarPos("ThreshI", "Controls", 47)  # inner
    cv2.createTrackbar("ThreshO", "Controls", 0, 255, _nothing)
    cv2.setTrackbarPos("ThreshO", "Controls", 69)  # outer


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
        inner_thresh = cv2.getTrackbarPos("ThreshI", "Controls")
        outer_thresh = cv2.getTrackbarPos("ThreshO", "Controls")

    _, threshI = cv2.threshold(inner, inner_thresh, 255, cv2.THRESH_BINARY)
    _, threshO = cv2.threshold(outer, outer_thresh, 255, cv2.THRESH_BINARY)

    thresh = cv2.add(threshI, threshO)
    return thresh


def _detect_centroids(contours: np.ndarray, min_area: int, max_area: int) -> list[Any]:
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
                centroids.append(Centroid(cX, cY))

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

    return centroids


####################################################################################################
# Public Functions                                                                                 #
####################################################################################################


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
    ret, frame = capture.read()

    retVal = DetectedCentroids([], [])

    if not ret:
        print("Error: Could not read frame.")
        exit()

    frame = _warp_image(frame)

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the background and current frame
    diff = cv2.absdiff(bg, gray_frame)

    retVal.fingers = _run_detection(diff, FINGER_PARAMS)
    retVal.cds = _run_detection(diff, CD_PARAMS)

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

    cv_init(True, True)

    while True:
        cv_loop()


if __name__ == "__main__":
    main()
