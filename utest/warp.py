from typing import Any

import cv2
import numpy as np

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

CFG_SHOW_INITIAL_CROP: bool = True
CFG_SHOW_BG_SUBTRACT: bool = False

# Currently required to be True due to
# using CV for key input to capture bg.
CFG_SHOW_INITIAL_BG: bool = True

# Only affects the non-standalone case.
# if standalone, this is always displayed.
CFG_SHOW_FRAME: bool = False

standalone: bool = False
capture: cv2.VideoCapture = cv2.VideoCapture()
bg: np.ndarray = np.zeros(0)

def warpImage(image, corners, target, width, height):
    corners_np = np.array(corners, dtype=np.float32)
    target_np = np.array(target, dtype=np.float32)
    
    mat = cv2.getPerspectiveTransform(corners_np, target_np)
    out = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_CUBIC)
    while True:
        cv2.namedWindow("out", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "out", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.imshow("out",out)
        cv2.waitKey(1)
    return out

def _capture_bg(capture: cv2.VideoCapture) -> np.ndarray:
    global bg

    print("Press 'b' to capture the background frame.")
    capturing: bool = True
    while capturing:
        ret, frame = capture.read()
        frame = frame[CROP]
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
    global CROP

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, CROP_MIN_THRESH, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # largest contour is the table
    if len(contours):
        table_outline = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(table_outline)
        print("PRE",x,y,w,h)
        y, h = (int(val * CROP_SCALE) for val in [y, h])
        print("POST", x,y,w,h)
        
        #x += X_OFFSET
        #w -= 2*X_OFFSET
        #y += Y_OFFSET
        #h -= 2*Y_OFFSET

        

        peri = cv2.arcLength(table_outline, True)
        corners = cv2.approxPolyDP(table_outline, 0.04 * peri, True)

        print(type(corners), corners[0])
        cv2.polylines(frame, [corners], True, (0,0,255), 1, cv2.LINE_AA)


        corners2 = [corners[0][0][0], corners[0][0][1],
                    corners[1][0][0], corners[1][0][1],
                    corners[2][0][0], corners[2][0][1],
                    corners[3][0][0], corners[3][0][1],]
        
        width = 1400
        height = 1050

        target = [(0,0),(width,0),(width,height),(0,height)]
        out = warpImage(frame, corners, target, width,height)
        
        print(corners2)



        

        CROP = (slice(y, y + h), slice(x, x + w))

    if CFG_SHOW_INITIAL_CROP:
        crop = frame[CROP]
        cv2.imshow("frame", frame)
        # cv2.imshow("crop", crop)
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


def cv_loop() -> list[Any]:
    ret, frame = capture.read()

    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        exit()

    frame = frame[CROP]

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between the background and current frame
    diff = cv2.absdiff(bg, gray_frame)

    # Apply a threshold to get a binary image
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

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

        # Filter out very small/large contours (adjust the threshold as needed)
        if CONTOUR_MIN_AREA < area and area < CONTOUR_MAX_AREA:
            # Calculate moments for each contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the centroid on the frame
                centroids.append([cX, cY])
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    # Display original frame with detected centroids and the threshold image
    if standalone or CFG_SHOW_FRAME:
        cv2.namedWindow("Detected Centroids", cv2.WND_PROP_FULLSCREEN)

        cv2.setWindowProperty(
            "Detected Centroids", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        cv2.imshow("Detected Centroids", cv2.flip(frame, 1))

    if CFG_SHOW_BG_SUBTRACT:
        cv2.imshow("Foreground (Background Subtraction)", thresh)

    # Exit the loop when 'q' is pressed
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
