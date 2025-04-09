import cv2
import numpy as np

def update_contours(margin):
    global current_margin
    current_margin = margin

cv2.namedWindow('Contours')
current_margin = 10
height= width = 500
cv2.createTrackbar('Margin', 'Contours', 10, min(height, width) // 2, update_contours)

#special gif case - take out for videos.
while True:
    cap = cv2.VideoCapture("giphy (1).gif")
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        cap.release()
        exit()
    height, width = frame.shape[:2]


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask with the updated margin
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (current_margin, current_margin), (width-current_margin, height-current_margin), 255, -1)
        masked_image = cv2.bitwise_and(gray, gray, mask=mask)

        # Threshold the masked image
        ret, thresh = cv2.threshold(masked_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = frame.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', contour_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('o'):  # Up arrow key
            new_margin = min(current_margin + 1, min(height, width) // 2)
            update_contours(new_margin)
            cv2.setTrackbarPos('Margin', 'Contours', current_margin)