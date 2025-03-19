import cv2
import numpy as np

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
            # Convert to grayscale for easier subtraction
            background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Background captured.")
            break
    cv2.destroyWindow("Live Feed - Press 'b' to capture background")
    return background

def main():
    # Open the default camera (0). Adjust if necessary.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Capture the background scene for subtraction
    background = capture_background(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference between the background and current frame
        diff = cv2.absdiff(background, gray_frame)
        
        # Apply a threshold to get a binary image
        _, thresh = cv2.threshold(diff, 65, 255, cv2.THRESH_BINARY)
        
        # Optionally, use morphological operations to remove small noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Loop over the contours to detect and draw centroids
        for cnt in contours:
            # Filter out very small contours (adjust the threshold as needed)
            if cv2.contourArea(cnt) < 15:
                continue
            
            # Calculate moments for each contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(frame, "centroid", (cX - 25, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the original frame with detected centroids and the threshold image
        cv2.imshow("Detected Centroids", frame)
        cv2.imshow("Foreground (Background Subtraction)", thresh)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
