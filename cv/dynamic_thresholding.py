import cv2
import numpy as np

# Configuration Options
BASELINE_THRESHOLD_VALUE = 30   # Constant offset for the threshold (tune this for sensitivity)
THRESHOLD_SCALE = 0.001           # Scaling factor for the baseline values to adapt the threshold
MIN_CONTOUR_AREA = 50          # Minimum contour area (in pixels) to be considered a valid finger press

def capture_baseline(cap):
    """
    Capture an initial snapshot of the table to store as the baseline light intensities.
    Press 'b' to capture the baseline snapshot.
    """
    print("Press 'b' to capture the baseline snapshot of the table.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            exit()
        cv2.imshow("Live Feed - Press 'b' for Baseline", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # Convert to grayscale for simpler intensity processing
            baseline = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Baseline captured.")
            break
    cv2.destroyWindow("Live Feed - Press 'b' for Baseline")
    return baseline

def process_frame(baseline, frame):
    """
    Process the current frame by comparing it to the baseline snapshot.
    A per-pixel dynamic threshold is computed using the baseline values.
    Contours corresponding to finger presses are then extracted.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the pixel-wise difference: positive difference means the current
    # pixel is darker than the baseline (adjust as needed if finger press brightens instead)
    diff = cv2.subtract(baseline, gray)
    
    # Compute the dynamic threshold per pixel:
    #   For each pixel: dynamic_threshold = BASELINE_THRESHOLD_VALUE + THRESHOLD_SCALE * baseline_pixel_value
    dynamic_threshold = BASELINE_THRESHOLD_VALUE + (THRESHOLD_SCALE * baseline)
    
    # Create a binary mask: pixels where the difference is greater than the dynamic threshold are considered pressed.
    # Convert the result to 255 (white) for a binary image.
    mask = (diff > dynamic_threshold).astype(np.uint8) * 255
    
    # Optional: Clean up the mask with morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Capture the baseline snapshot from the table
    baseline = capture_baseline(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame using the baseline snapshot
        mask = process_frame(baseline, frame)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and centroids for detected finger presses
        output = frame.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue

            # Draw the contour
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            
            # Compute and draw the centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(output, "centroid", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show the binary mask and the output with detected contours/centroids
        cv2.imshow("Dynamic Threshold Mask", mask)
        cv2.imshow("Detected Finger Presses", output)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np

# Configuration Options
BASELINE_THRESHOLD_VALUE = 30   # Constant offset for the threshold (tune this for sensitivity)
THRESHOLD_SCALE = 0.1           # Scaling factor for the baseline values to adapt the threshold
MIN_CONTOUR_AREA = 100          # Minimum contour area (in pixels) to be considered a valid finger press

def capture_baseline(cap):
    """
    Capture an initial snapshot of the table to store as the baseline light intensities.
    Press 'b' to capture the baseline snapshot.
    """
    print("Press 'b' to capture the baseline snapshot of the table.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            exit()
        cv2.imshow("Live Feed - Press 'b' for Baseline", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # Convert to grayscale for simpler intensity processing
            baseline = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Baseline captured.")
            break
    cv2.destroyWindow("Live Feed - Press 'b' for Baseline")
    return baseline

def process_frame(baseline, frame):
    """
    Process the current frame by comparing it to the baseline snapshot.
    A per-pixel dynamic threshold is computed using the baseline values.
    Contours corresponding to finger presses are then extracted.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the pixel-wise difference: positive difference means the current
    # pixel is darker than the baseline (adjust as needed if finger press brightens instead)
    diff = cv2.subtract(baseline, gray)
    
    # Compute the dynamic threshold per pixel:
    #   For each pixel: dynamic_threshold = BASELINE_THRESHOLD_VALUE + THRESHOLD_SCALE * baseline_pixel_value
    dynamic_threshold = BASELINE_THRESHOLD_VALUE + (THRESHOLD_SCALE * baseline)
    
    # Create a binary mask: pixels where the difference is greater than the dynamic threshold are considered pressed.
    # Convert the result to 255 (white) for a binary image.
    mask = (diff > dynamic_threshold).astype(np.uint8) * 255
    
    # Optional: Clean up the mask with morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Capture the baseline snapshot from the table
    baseline = capture_baseline(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame using the baseline snapshot
        mask = process_frame(baseline, frame)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and centroids for detected finger presses
        output = frame.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue

            # Draw the contour
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            
            # Compute and draw the centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(output, "centroid", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show the binary mask and the output with detected contours/centroids
        cv2.imshow("Dynamic Threshold Mask", mask)
        cv2.imshow("Detected Finger Presses", output)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
