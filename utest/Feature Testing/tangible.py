from typing import Any

import cv2
import numpy as np

import timeit

# Run `ls /dev | grep video` to see which idx to use for the camera
CAM_IDX: int = 0

# USAGE: frame[CROP]         Y1:Y2      , X1:X2
CROP: tuple[slice, slice] = (slice(0, 1), slice(0, 1))

CROP_SCALE: float = 0.975
CROP_MIN_THRESH: int = 40

X_OFFSET: int = 25
Y_OFFSET: int = 15

CONTOUR_MIN_AREA: int = 35
CONTOUR_MAX_AREA: int = 1000

CFG_SHOW_INITIAL_CROP: bool = False
CFG_SHOW_BG_SUBTRACT: bool = False

# Currently required to be True due to using CV for key input to capture bg.
CFG_SHOW_INITIAL_BG: bool = False

# Only affects the non-standalone case.
# if standalone, this is always displayed.
CFG_SHOW_FRAME: bool = False

WIDTH = 1400
HEIGHT = 1050
TARGET = [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]

corners: np.ndarray = np.zeros(0)
standalone: bool = False
capture: cv2.VideoCapture = cv2.VideoCapture()
bg: np.ndarray = np.zeros(0)
transformed: bool = False
mat: cv2.typing.MatLike
current_margin:int=0
acctime = 0
frames = 0


def update_contours(margin):
    global current_margin
    current_margin = margin

# -------------------------------------------------------------------------
# New configuration options for adaptive thresholding based on edge distance:
CFG_CENTER_THRESHOLD: int = 20+8           # Base threshold (applied at the center, farthest from any edge)
CFG_THRESHOLD_DISTANCE_SCALE: float = 30.0-8 # Additional threshold applied at the edges
# New configuration options for drag thresholding
CFG_DRAG_RADIUS: int = 50      # Radius (in pixels) around a detected centroid to lighten the threshold
CFG_LIGHTEN_AMOUNT: int = 20   # Amount by which to lower the threshold in the drag area
# -------------------------------------------------------------------------

# Global variable to track the active centroid position (if any)
active_centroid: list[int] | None = None

def warpImage(image: np.ndarray) -> np.ndarray:
    global transformed, mat
    corners_np = np.array(corners, dtype=np.float32)
    target_np = np.array(TARGET, dtype=np.float32)
    if (not transformed):
        mat = cv2.getPerspectiveTransform(corners_np, target_np)
        transformed=True
    out = cv2.warpPerspective(image, mat, (WIDTH, HEIGHT), flags=cv2.INTER_CUBIC)
    return out

def _capture_bg(capture: cv2.VideoCapture) -> np.ndarray:
    global bg

    print("Press 'b' to capture the background frame.")
    capturing: bool = True
    while capturing:
        ret, frame = capture.read()
        frame = warpImage(frame)
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

def _crop_bg(frame: np.ndarray) -> None:
    global corners

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, CROP_MIN_THRESH, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        table_outline = max(contours, key=cv2.contourArea)    
        peri = cv2.arcLength(table_outline, True)
        approx = cv2.approxPolyDP(table_outline, 0.04 * peri, True)

        if len(approx) == 4:  # Ensure exactly 4 points are found
            corners = approx.reshape(4, 2)
        else:
            print("Warning: Could not find exactly 4 corner points. Using default corners.")
            corners = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype=np.float32)

        cv2.polylines(frame, [corners.astype(int)], True, (0, 0, 255), 1, cv2.LINE_AA)

    else:
        print("Warning: No contours detected for cropping. Using default corners.")
        corners = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype=np.float32)

    if CFG_SHOW_INITIAL_CROP:
        cv2.imshow("frame", frame)
        cv2.waitKey(2000)

def cv_init() -> None:
    global capture

    capture = cv2.VideoCapture(CAM_IDX)
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

# Helper to ensure block size is odd and >= 3
def make_odd(val):
    return val if val % 2 == 1 else val + 1
# Trackbar callback (does nothing; we use getTrackbarPos)
def nothing(x):
    pass

skipframe: int = 20
cnt:int = 0

def cv_loop() -> list[Any]:
    #global acctime, frames
    start = timeit.default_timer()
    global cnt
    block_size = cv2.getTrackbarPos("Block Size", "Controls")
    C = cv2.getTrackbarPos("C", "Controls")
    lowerthresh= cv2.getTrackbarPos("Lower Thresh", "Controls")


    block_size = make_odd(max(3, block_size))  # Ensure block size is odd and >=3

    global active_centroid, current_margin

    ret, frame = capture.read()
    '''if cnt != skipframe:
        cnt += 1
        return'''
    

    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        exit()
    frame = warpImage(frame)

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compute absolute difference between the background and current frame
    diff = cv2.absdiff(bg, gray_frame)

    #Mask margin
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (current_margin, current_margin), (width-current_margin, height-current_margin), 255, -1)
    diff = cv2.bitwise_and(diff, mask)

    # Apply adaptive threshold
    #ret2,diff = cv2.threshold(diff,lowerthresh,255,cv2.THRESH_BINARY)
    #ret, diff = cv2.threshold(diff, lowerthresh, 255, cv2.THRESH_BINARY)
    diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    
    # Use morphological operations to remove small noise
    '''kernel = np.ones((3, 3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)'''

    '''kernel = np.ones((3,3),np.uint8)
    diff = cv2.erode(diff,kernel,iterations = 1)'''
    '''diff = cv2.dilate(diff,kernel,iterations = 10)'''

    # Detect circles using HoughCircles
    #diff = cv2.medianBlur(diff, 7)
    #diff = cv2.GaussianBlur(diff, (7,7),0)

    circles = cv2.HoughCircles(
    diff,
    cv2.HOUGH_GRADIENT,
    dp=cv2.getTrackbarPos("dp", "Controls")/10, 
    minDist=cv2.getTrackbarPos("minDist", "Controls"),
    param1=cv2.getTrackbarPos("param1", "Controls"),
    param2=cv2.getTrackbarPos("param2", "Controls") /10,   # Increase if detecting too many false positives
    minRadius=cv2.getTrackbarPos("minRadius", "Controls"),
    maxRadius=cv2.getTrackbarPos("maxRadius", "Controls")
    )


    # Draw circles if any were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 155, 0), 2)   # outer circle
            cv2.circle(frame, (x, y), 2, (0, 155, 0), 3)   # center dot
    


    

    # Find contours
    #contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    '''for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 255, 0), 2)  # green rectangle'''

    # Filter based on area and aspect ratio to get the rectangular piece
    '''for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        #print("polyDP:", approx)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(diff, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            img = cv2.drawContours(diff, [cnt], -1, (0,255,0), 3)'''



    # Display original frame with detected centroids and the threshold image
    if standalone or CFG_SHOW_FRAME:
        cv2.namedWindow("Detected Centroids", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Detected Centroids", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Detected Centroids", frame)
        '''cv2.namedWindow("Diff", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Diff", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Diff", diff)'''


    stop = timeit.default_timer()
    #frames+=1
    #acctime += stop-start
    print(stop-start)

    # Exit the loop when 'q' is pressed
    if standalone and cv2.waitKey(1) == ord("q"):
        capture.release()
        cv2.destroyAllWindows()
        print("mean time: ", frames/acctime)
        exit()

    centroids = []
    return centroids

def main() -> None:
    global standalone
    standalone = True

    cv_init()
    
    ##Control Panel Window!
    cv2.namedWindow("Controls", cv2.WND_PROP_FULLSCREEN)
    ret, frame = capture.read()
    height, width = frame.shape[:2] #may to readjust this?
    cv2.createTrackbar('Margin', "Controls", 10, min(height, width) // 2, update_contours)
    cntrl = cv2.imread("control.png")
    cv2.imshow("Controls", cntrl)

    cv2.createTrackbar("Block Size", "Controls", 11, 50, nothing)
    cv2.createTrackbar("C", "Controls", 2, 20, nothing)
    cv2.createTrackbar("Lower Thresh", "Controls", 0, 255, nothing)
    cv2.setTrackbarPos('Lower Thresh', 'Controls', 18)
    cv2.setTrackbarPos('Block Size', 'Controls', 32)
    cv2.setTrackbarPos('C', 'Controls', 6)
    ##Hough Circles
    cv2.createTrackbar("dp", "Controls", 10, 30, nothing)
    cv2.createTrackbar("minDist", "Controls", 0, 255, nothing)
    cv2.createTrackbar("param1", "Controls", 0, 1200, nothing)
    cv2.createTrackbar("param2", "Controls", 1, 300, nothing)
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