import cv2
import numpy as np
import copy
import time

CFG_SHOW_INITIAL_CROP=False
SHOW_FPS=False

### Globals ###
WIDTH = 1400
HEIGHT = 1050
TARGET = [(0,0),(WIDTH,0),(WIDTH,HEIGHT),(0,HEIGHT)]
corners: np.ndarray = np.zeros(0)
standalone: bool = False
capture: cv2.VideoCapture = cv2.VideoCapture()
bg: np.ndarray = np.zeros(0)

# Define a function to reorder rectangle corners
def _reorder_corners(corners: np.ndarray) -> np.ndarray:
    threshold = 200
    new_corners = copy.deepcopy(corners)
    for cor in corners: # Strange ordering to line up with table
        if cor[0][0] < threshold and cor[0][1] < threshold:
            new_corners[1] = cor # LOW, LOW
        elif cor[0][0] > threshold and cor[0][1] < threshold:
             new_corners[0] = cor # BIG, LOW
        elif cor[0][0] > threshold and cor[0][1] > threshold:
             new_corners[3] = cor # BIG, BIG
        else:
             new_corners[2] = cor # LOW, BIG
    return new_corners


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

def nothing(x):
    pass

# Function to open the webcam and show the video feed
def read_webcam(camera_index=0, width=640, height=480, show_fps=False):
    global corners
    cv2.namedWindow("Controls", cv2.WND_PROP_FULLSCREEN)
    cv2.createTrackbar('Margin', "Controls", 10, min(height, width) // 2, nothing)
    cntrl = cv2.imread("control.png")
    cv2.imshow("Controls", cntrl)
    cv2.createTrackbar("Cam", "Controls", 0, 400, nothing)
    cv2.setTrackbarPos('Cam', 'Controls', 28) ##inner
    
    # Open a connection to the webcam
    cap = cv2.VideoCapture(camera_index)

    # Set the desired width and height if supported by the camera
    '''capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    

    capture.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    capture.set(cv2.CAP_PROP_CONTRAST, 10)
    capture.set(cv2.CAP_PROP_SATURATION, 0)
    capture.set(cv2.CAP_PROP_SHARPNESS, 0)
    
    capture.set(cv2.CAP_PROP_BACKLIGHT, 0)
    capture.set(cv2.CAP_PROP_ZOOM, 0)
    capture.set(cv2.CAP_PROP_EXPOSURE,-8)
    capture.set(cv2.CAP_PROP_PAN,0)
    capture.set(cv2.CAP_PROP_TILT,0)
    time.sleep(3)'''

    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Main loop to capture frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print(frame.shape[:2])

        if not ret:
            print("Failed to capture image")
            break

        # If show_fps is True, calculate and display FPS
        if SHOW_FPS:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # largest contour is the table
        if len(contours):
            table_outline = max(contours, key=cv2.contourArea)    

            peri = cv2.arcLength(table_outline, True)
            corners = cv2.approxPolyDP(table_outline, 0.04 * peri, True)
            #print(corners)
            #corners = _reorder_corners(corners)
            #print("rearranged:", corners)
            cv2.polylines(frame, [corners], True, (0,0,255), 1, cv2.LINE_AA)


        # Display the resulting frame
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Webcam", cv2.WND_PROP_FULLSCREEN, 1
        )
        cv2.imshow('Webcam', (frame))


        

        

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Example function call with customization options
read_webcam(camera_index=0, width=1280, height=720, show_fps=True)