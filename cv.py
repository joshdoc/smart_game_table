from typing import Any

import cv2
import numpy as np
import time
import copy

### Constants ###
# Run `ls /dev | grep video` to see which idx to use for the camera
CAM_IDX: int = 0

# USAGE: frame[CROP]         Y1:Y2      , X1:X2
CROP: tuple[slice, slice] = (slice(0, 1), slice(0, 1))
CROP_SCALE: float = 0.975
CROP_MIN_THRESH: int = 30

X_OFFSET: int = 25
Y_OFFSET: int = 15

CONTOUR_MIN_AREA: int = 35
CONTOUR_MAX_AREA: int = 1000

WIDTH = 1400
HEIGHT = 1050
TARGET = [(0,0),(WIDTH,0),(WIDTH,HEIGHT),(0,HEIGHT)]

current_margin:int=0
def update_contours(margin):
    global current_margin
    current_margin = margin
def nothing(x):
    pass

# Adaptive | Rectangular #
CENTER_THRESHOLD_RECT: int = 20+8           # Base threshold (applied at the center, farthest from any edge)
THRESHOLD_DISTANCE_SCALE_RECT: float = 30.0-8 # Additional threshold applied at the edges

# Adaptive | From Center #
adj = 15
CENTER_THRESHOLD: int = 25+adj              # Base threshold at the center of the image
THRESHOLD_DISTANCE_SCALE: float = 25.0-10  # How much the threshold increases at the maximum distance

### Configuration Options ###
CFG_SHOW_INITIAL_CROP: bool = False
CFG_SHOW_BG_SUBTRACT: bool = False
CFG_SHOW_INITIAL_BG: bool = False
CFG_SHOW_FRAME: bool = False
CFG_LOG_TIME: bool = False
# Change mode to adaptive
CFG_ADAPTIVE: bool = False
CFG_ADAPTIVE_RECT: bool = False # overrides adaptive if both true

ALPHA = .5
BETA = 1 - ALPHA  # Inverse transparency factor for the background
CUT_LOW = 15
CUT_RIGHT = 5
CUT_LEFT = 5
CUT_TOP = 10

### Globals ###
corners: np.ndarray = np.zeros(0)
standalone: bool = False
capture: cv2.VideoCapture = cv2.VideoCapture()
bg: np.ndarray = np.zeros(0)

fps = 0
frame_count = 0
start_time = time.time()

### Define a function to reorder rectangle corners
def _reorder_corners(corners: np.ndarray) -> np.ndarray:
    threshold = 200
    new_corners = copy.deepcopy(corners)
    for cor in corners: # Strange ordering to line up with table
        if cor[0][0] < threshold and cor[0][1] < threshold:
            new_corners[1] = cor
        elif cor[0][0] > threshold and cor[0][1] < threshold:
             new_corners[0] = cor
        elif cor[0][0] > threshold and cor[0][1] > threshold:
             new_corners[3] = cor
        else:
             new_corners[2] = cor
    new_corners[2][0][1] -= CUT_LOW
    new_corners[3][0][1] -= CUT_LOW
    new_corners[0][0][0] -= CUT_RIGHT
    new_corners[3][0][0] -= CUT_RIGHT
    new_corners[1][0][0] -= CUT_LEFT
    new_corners[2][0][0] -= CUT_LEFT
    new_corners[2][0][1] += CUT_TOP
    new_corners[3][0][1] += CUT_TOP
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
        cv2.polylines(frame, [corners], True, (0,0,255), 1, cv2.LINE_AA)
        corners = _reorder_corners(corners)
        #out = _warp_image(frame)   

    if CFG_SHOW_INITIAL_BG:
        cv2.imshow("frame", frame)
        cv2.waitKey(2000)


### Correct camera perspective to match table top
def _warp_image(image: np.ndarray) -> np.ndarray:
    corners_np = np.array(corners, dtype=np.float32)
    target_np = np.array(TARGET, dtype=np.float32)
    
    mat = cv2.getPerspectiveTransform(corners_np, target_np)
    out = cv2.warpPerspective(image, mat, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
    while CFG_SHOW_INITIAL_CROP:
        cv2.namedWindow("out", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "out", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.imshow("out",out)
        cv2.waitKey(1)
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


def cv_init() -> None:
    global capture

    capture = cv2.VideoCapture(CAM_IDX)
    # Set the desired width and height if supported by the camera
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    

    '''capture.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    capture.set(cv2.CAP_PROP_CONTRAST, 10)
    capture.set(cv2.CAP_PROP_SATURATION, 0)
    capture.set(cv2.CAP_PROP_SHARPNESS, 0)
    
    capture.set(cv2.CAP_PROP_BACKLIGHT, 0)
    capture.set(cv2.CAP_PROP_ZOOM, 0)
    capture.set(cv2.CAP_PROP_EXPOSURE,-8)
    capture.set(cv2.CAP_PROP_PAN,0)
    capture.set(cv2.CAP_PROP_TILT,0)
    time.sleep(3)'''
    #capture.set(cv2.CAP_PROP_FPS,120)
    #capture.set(cv2.CAP_PROP_EXPOSURE,0)

    if not capture.isOpened():
        print("Error: Could not open the camera.")
        exit()

    # Automatically crop image
    ret, frame = capture.read()

    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        exit()

    _crop_bg(frame)

    # Capture the background scene for subtraction
    _capture_bg(capture)

def make_odd(val):
    return val if val % 2 == 1 else val + 1

### Detect centroids (finger presses) and return list
def cv_loop() -> list[Any]:
    global fps, frame_count, start_time
    ret, frame = capture.read()

    block_size = cv2.getTrackbarPos("Block Size", "Controls")
    block_size = make_odd(max(3, block_size))
    C = cv2.getTrackbarPos("C", "Controls")

    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        exit()

    if CFG_LOG_TIME:
        start = timeit.default_timer()

    frame = _warp_image(frame)

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the background and current frame
    #circle_diff = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    #circle 42/42
    diff = cv2.absdiff(bg, gray_frame)

    #Inner Rectangle (Sens ~24?) / Outer (Sens ~35)
    topLX = 313
    topLY = 198
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (topLX, topLY), (width-topLX, height-topLY), 255, -1)
    innie = cv2.bitwise_and(diff, mask)
    outie = cv2.bitwise_and(diff,cv2.bitwise_not(mask))

    # Different thresholds for the different sections
    _, threshI = cv2.threshold(innie, cv2.getTrackbarPos("ThreshI", "Controls"), 255, cv2.THRESH_BINARY)
    _, threshO = cv2.threshold(outie, cv2.getTrackbarPos("ThreshO", "Controls"), 255, cv2.THRESH_BINARY)
    #_, threshI = cv2.threshold(innie, 47, 255, cv2.THRESH_BINARY)
    #_, threshO = cv2.threshold(outie, 69, 255, cv2.THRESH_BINARY)
    thresh = cv2.add(threshI,threshO)

    # Use morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find all contours in the thresholded image
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop over the contours to detect and draw centroids
    #CONTOUR_MIN_AREA = cv2.getTrackbarPos("ConMinArea", "Controls")

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        #if solidity > 0.8 and hull_area<1500 and hull_area > 100:
        cv2.drawContours(frame, [hull], 0, (255, 255, 0), 2)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw the centroid on the frame
            txt = "Solidity:" + str(solidity) 
            txt2="Area: " + str(hull_area)
            cv2.putText(frame, txt , (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, txt2 , (cX, cY+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            centroids.append([cX, cY])
            cv2.circle(frame, (cX, cY), 5, (0, 255, 255), -1)
       
        
    '''circles = cv2.HoughCircles(
    circle_diff,
    cv2.HOUGH_GRADIENT,
    dp=cv2.getTrackbarPos("dp", "Controls")/10, 
    minDist=cv2.getTrackbarPos("minDist", "Controls"),
    param1=cv2.getTrackbarPos("param1", "Controls"),
    param2=cv2.getTrackbarPos("param2", "Controls") /10,   # Increase if detecting too many false positives
    minRadius=cv2.getTrackbarPos("minRadius", "Controls"),
    maxRadius=cv2.getTrackbarPos("maxRadius", "Controls")
    )'''


    # Draw circles if any were found
    '''if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 155, 0), 2)   # outer circle
            cv2.circle(frame, (x, y), 2, (0, 155, 0), 3)   # center dot'''

     # Calculate FPS every second
    current_time = time.time()  # Current time
    elapsed_time = current_time - start_time  # Time elapsed since the last FPS calculation

    if elapsed_time > 1:  # If more than 1 second has passed
        fps = frame_count / elapsed_time  # Calculate FPS
        #print("FPS:", fps)
        start_time = current_time  # Reset start time
        frame_count = 0  # Reset frame counter

    # Display the FPS on the frame
    fps_display = f"FPS: {fps:.2f}"
    cv2.putText(thresh, fps_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, fps_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frame_count += 1  # Increment frame counter

    # Display original frame with detected centroids and the threshold image
    if standalone or CFG_SHOW_FRAME:
        cv2.namedWindow("Detected Centroids", cv2.WINDOW_NORMAL)

        cv2.setWindowProperty(
            "Detected Centroids", cv2.WND_PROP_FULLSCREEN, 1
        )

        # Perform weighted addition
        # Convert the single-channel image1 to a three-channel image
        t_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        overlayed_image = cv2.addWeighted(t_colored, ALPHA, frame, BETA, 0)
        cv2.imshow("Detected Centroids", overlayed_image)
        #cv2.imshow("Detected Centroids", thresh)

    if CFG_SHOW_BG_SUBTRACT:
        cv2.imshow("Foreground (Background Subtraction)", thresh)

    if CFG_LOG_TIME:
        #stop = timeit.default_timer()
        print("elapsed time: ")

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

    ##Control Panel Window!
    
    ret, frame = capture.read()
    height, width = frame.shape[:2] #may to readjust this?
    
    cv2.namedWindow("Controls", cv2.WND_PROP_FULLSCREEN)
    cv2.createTrackbar('Margin', "Controls", 10, min(height, width) // 2, update_contours)
    cntrl = cv2.imread("debug/control.png")
    cv2.imshow("Controls", cntrl)

    cv2.createTrackbar("ThreshI", "Controls", 0, 255, nothing)
    cv2.setTrackbarPos('ThreshI', 'Controls', 47) ##inner
    cv2.createTrackbar("ThreshO", "Controls", 0, 255, nothing)
    cv2.setTrackbarPos('ThreshO', 'Controls', 69) ##outer
    cv2.createTrackbar("ConMinArea", "Controls", 0, 120, nothing)
    cv2.setTrackbarPos('ConMinArea', 'Controls', 35) 
    cv2.createTrackbar("CameraSet", "Controls", 0, 10, nothing)
    cv2.setTrackbarPos('CameraSet', 'Controls', 0) 

    cv2.createTrackbar("Block Size", "Controls", 11, 50, nothing)
    cv2.createTrackbar("C", "Controls", 1, 20, nothing)
    cv2.createTrackbar("Lower Thresh", "Controls", 0, 255, nothing)
    cv2.setTrackbarPos('Lower Thresh', 'Controls', 18)
    cv2.setTrackbarPos('Block Size', 'Controls', 31)
    cv2.setTrackbarPos('C', 'Controls', 5)

    #CD threshlding - remove later
    cv2.setTrackbarPos('ThreshI', 'Controls', 42) ##inner
    cv2.setTrackbarPos('ThreshO', 'Controls', 42) ##outer
   
    ##Hough Circles Controls
    cv2.createTrackbar("dp", "Controls", 10, 30, nothing)
    cv2.createTrackbar("minDist", "Controls", 0, 255, nothing)
    cv2.createTrackbar("param1", "Controls", 0, 1200, nothing)
    cv2.createTrackbar("param2", "Controls", 1, 800, nothing)
    cv2.createTrackbar("minRadius", "Controls", 0, 255, nothing)
    cv2.createTrackbar("maxRadius", "Controls", 0, 255, nothing)
    cv2.setTrackbarPos("dp", "Controls", 10)
    cv2.setTrackbarPos("minDist", "Controls", 190)
    cv2.setTrackbarPos("param1", "Controls", 352)
    cv2.setTrackbarPos("param2", "Controls", 156)
    cv2.setTrackbarPos("minRadius", "Controls", 124)
    cv2.setTrackbarPos("maxRadius", "Controls", 150)

    while True:
        cv_loop()


if __name__ == "__main__":
    main()
