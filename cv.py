import copy
import time
from typing import Any

import cv2
import numpy as np

####################################################################################################
# Constants                                                                                        #
####################################################################################################
# Run `ls /dev | grep video` to see which idx to use for the camera (or guess on windows)
CAM_IDX: int = 0

# USAGE: frame[CROP]         Y1:Y2      , X1:X2
CROP: tuple[slice, slice] = (slice(0, 1), slice(0, 1))

CROP_SCALE: float = 0.975  # scale the cropped image
CROP_MIN_THRESH: int = 30  # threshold for detecting the edges

FINGER_MIN_AREA: int = 35  # TODO: use these constants
FINGER_MAX_AREA: int = 1000
  
# define screen size for warping the image
WIDTH = 1400
HEIGHT = 1050
TARGET = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]

# offsets for correcting warp errors
CUT_LOW = 15
CUT_RIGHT = 5
CUT_LEFT = 5
CUT_TOP = -10

# transparency settings for multi-rectangle thresholding
ALPHA = 0.5
BETA = 1 - ALPHA

# Adaptive | Distance to edge (rect) #
# Base threshold (applied at the center, farthest from any edge)
CENTER_THRESHOLD_RECT: int = 28
# Additional threshold applied at the edges
THRESHOLD_DISTANCE_SCALE_RECT: float = 22.0

# Adaptive | Distance to center #
# Base threshold at the center of the image
CENTER_THRESHOLD: int = 40
# How much the threshold increases at the maximum distance
THRESHOLD_DISTANCE_SCALE: float = 15.0

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
CFG_USE_TRACKBARS: bool = True

# Change mode to adaptive
CFG_ADAPTIVE: bool = False
CFG_ADAPTIVE_RECT: bool = False  # overrides adaptive if both true


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
    cv2.createTrackbar(
        "Margin", "Controls", 10, min(height, width) // 2, _update_contours
    )

    control_image = cv2.imread("debug/control.png")
    cv2.imshow("Controls", control_image)

    cv2.createTrackbar("ThreshI", "Controls", 0, 255, _nothing)
    cv2.setTrackbarPos("ThreshI", "Controls", 47)  # inner
    cv2.createTrackbar("ThreshO", "Controls", 0, 255, _nothing)
    cv2.setTrackbarPos("ThreshO", "Controls", 69)  # outer
    cv2.createTrackbar("ConMinArea", "Controls", 0, 120, _nothing)
    cv2.setTrackbarPos("ConMinArea", "Controls", 35)
    cv2.createTrackbar("CameraSet", "Controls", 0, 10, _nothing)
    cv2.setTrackbarPos("CameraSet", "Controls", 0)

    cv2.createTrackbar("Block Size", "Controls", 11, 50, _nothing)
    cv2.createTrackbar("C", "Controls", 1, 20, _nothing)
    cv2.createTrackbar("Lower Thresh", "Controls", 0, 255, _nothing)
    cv2.setTrackbarPos("Lower Thresh", "Controls", 18)
    cv2.setTrackbarPos("Block Size", "Controls", 31)
    cv2.setTrackbarPos("C", "Controls", 5)

    # CD threshlding - remove later
    cv2.setTrackbarPos("ThreshI", "Controls", 42)  # inner
    cv2.setTrackbarPos("ThreshO", "Controls", 42)  # outer

    # Hough Circles Controls
    cv2.createTrackbar("dp", "Controls", 10, 30, _nothing)
    cv2.createTrackbar("minDist", "Controls", 0, 255, _nothing)
    cv2.createTrackbar("param1", "Controls", 0, 1200, _nothing)
    cv2.createTrackbar("param2", "Controls", 1, 800, _nothing)
    cv2.createTrackbar("minRadius", "Controls", 0, 255, _nothing)
    cv2.createTrackbar("maxRadius", "Controls", 0, 255, _nothing)
    cv2.setTrackbarPos("dp", "Controls", 10)
    cv2.setTrackbarPos("minDist", "Controls", 190)
    cv2.setTrackbarPos("param1", "Controls", 352)
    cv2.setTrackbarPos("param2", "Controls", 156)
    cv2.setTrackbarPos("minRadius", "Controls", 124)
    cv2.setTrackbarPos("maxRadius", "Controls", 150)

####################################################################################################
# Public Functions                                                                                 #
####################################################################################################
### Initialize the program
def cv_init() -> None:
    _capture_init()

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
def cv_loop() -> list[Any]:
    global fps, frame_count, start_time
    ret, frame = capture.read()

    '''block_size = cv2.getTrackbarPos("Block Size", "Controls")
    block_size = (max(3, block_size)) | 0b1
    C = cv2.getTrackbarPos("C", "Controls")'''

    if not ret:
        print("Error: Could not read frame.")
        exit()

    frame = _warp_image(frame)

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the background and current frame
    # circle_diff = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    # circle 42/42
    diff = cv2.absdiff(bg, gray_frame)

    # Inner Rectangle (Sens ~24?) / Outer (Sens ~35)
    topLX = 313
    topLY = 198
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (topLX, topLY), (width - topLX, height - topLY), 255, -1)
    inner = cv2.bitwise_and(diff, mask)
    outer = cv2.bitwise_and(diff, cv2.bitwise_not(mask))

    # Different thresholds for the different sections
    if CFG_USE_TRACKBARS:
        _, threshI = cv2.threshold(
            inner, cv2.getTrackbarPos("ThreshI", "Controls"), 255, cv2.THRESH_BINARY
        )
        _, threshO = cv2.threshold(
            outer, cv2.getTrackbarPos("ThreshO", "Controls"), 255, cv2.THRESH_BINARY
        )
    # TODO: make constants for these thresholds
    else:
        _, threshI = cv2.threshold(inner, 47, 255, cv2.THRESH_BINARY)
        _, threshO = cv2.threshold(outer, 69, 255, cv2.THRESH_BINARY)

    thresh = cv2.add(threshI, threshO)

    # Use morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop over the contours to detect and draw centroids
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        if (solidity > 0.8 and hull_area < 1500 and hull_area > 100) or (
            solidity > 0.8 and hull_area < 55000 and hull_area > 40000
        ):
            cv2.drawContours(frame, [hull], 0, (255, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the centroid on the frame
                txt = "Solidity:" + str(solidity)
                txt2 = "Area: " + str(hull_area)
                cv2.putText(
                    frame, txt, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    txt2,
                    (cX, cY + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
                centroids.append([cX, cY])
                cv2.circle(frame, (cX, cY), 5, (0, 255, 255), -1)

    current_time = time.time()
    elapsed_time = current_time - start_time

    # update FPS each second
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        # reset counters
        start_time = current_time
        frame_count = 0

    # Display the FPS on the frame
    fps_display = f"FPS: {fps:.2f}"
    cv2.putText(
        thresh, fps_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    cv2.putText(
        frame, fps_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    frame_count += 1  # Increment frame counter

    # Display original frame with detected centroids and the threshold image
    if standalone or CFG_SHOW_FRAME:
        cv2.namedWindow("Detected Centroids", cv2.WINDOW_NORMAL)

        cv2.setWindowProperty("Detected Centroids", cv2.WND_PROP_FULLSCREEN, 1)

        # Perform weighted addition
        # Convert the single-channel image1 to a three-channel image
        # t_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # overlayed_image = cv2.addWeighted(t_colored, ALPHA, frame, BETA, 0)
        # cv2.imshow("Detected Centroids", overlayed_image)
        cv2.imshow("Detected Centroids", frame)

    if CFG_SHOW_BG_SUBTRACT:
        cv2.imshow("Foreground (Background Subtraction)", thresh)

    # Exit the loop when 'q' is pressed if running standalone
    if standalone and cv2.waitKey(1) == ord("q"):
        capture.release()
        cv2.destroyAllWindows()
        exit()

    return centroids


def main() -> None:
    global standalone
    standalone = True

    cv_init()

    while True:
        cv_loop()


if __name__ == "__main__":
    main()
