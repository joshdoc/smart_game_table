import cv2
import numpy as np

# Configuration Options
MIN_CONTOUR_AREA = 100        # Minimum area (in pixels) to consider a region as a valid finger
CENTROID_RADIUS = 5           # Radius of the drawn centroid
THRESHOLD_METHOD = "adaptive" # Options: "adaptive" or "fixed"
FIXED_THRESHOLD_VALUE = 60    # Only used if THRESHOLD_METHOD == "fixed"
USE_INVERSION = True          # If True, invert the threshold output so fingers are white on black

# For adaptive thresholding
ADAPTIVE_BLOCK_SIZE = 11      # Must be odd and greater than 1
ADAPTIVE_C = 2                # Constant subtracted from the mean

# Morphological tuning
MORPH_KERNEL_SIZE = (7, 7)    # Kernel size for closing operations

def capture_background(cap):
    """
    Capture an initial background image.
    Press 'b' to capture and lock the background.
    """
    print("Press 'b' to capture the background frame.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            exit()
        cv2.imshow("Live Feed - Press 'b' to capture background", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # Convert to grayscale and apply Gaussian blur
            background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = cv2.GaussianBlur(background, (5, 5), 0)
            print("Background captured.")
            break
    cv2.destroyWindow("Live Feed - Press 'b' to capture background")
    return background

def process_frame(background, frame):
    """
    Process a single frame: perform background subtraction, thresholding, and morphological operations.
    Then, use the convex hull of each contour to fill in the finger region.
    """
    # Convert frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Compute absolute difference between background and current frame
    diff = cv2.absdiff(background, blurred_frame)
    
    # Apply thresholding (adaptive or fixed)
    if THRESHOLD_METHOD == "adaptive":
        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    else:
        _, thresh = cv2.threshold(diff, FIXED_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    
    # Optionally invert so fingers become white on a black background
    if USE_INVERSION:
        thresh = cv2.bitwise_not(thresh)
    
    # Use morphological closing to reduce noise and fill small gaps
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours from the closed image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to draw the filled convex hulls
    mask = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        # Compute the convex hull and draw it filled on the mask
        hull = cv2.convexHull(cnt)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    
    return mask

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Capture the background
    background = capture_background(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to obtain a binary mask with filled finger regions
        filled_mask = process_frame(background, frame)
        
        # Find contours on the filled mask for centroid detection
        contours, _ = cv2.findContours(filled_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw centroids on a copy of the original frame
        centroid_display = frame.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(centroid_display, (cX, cY), CENTROID_RADIUS, (0, 0, 255), -1)
                cv2.putText(centroid_display, "centroid", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display the filled regions and the centroid overlay
        cv2.imshow("Filled Finger Regions", filled_mask)
        cv2.imshow("Detected Centroids", centroid_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __
