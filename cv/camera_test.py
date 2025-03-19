import cv2
import numpy as np

def detect_centroids(frame, lower_thresh, upper_thresh):
    """
    Detect centroids of objects in the image based on thresholding values.

    :param frame: Input image frame.
    :param lower_thresh: Lower threshold value for binary thresholding.
    :param upper_thresh: Upper threshold value for binary thresholding.
    :return: List of centroids (x, y) positions.
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, lower_thresh, upper_thresh, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    # Loop over the contours and calculate the centroids
    for contour in contours:
        # Only consider contours with area greater than a threshold
        if cv2.contourArea(contour) > 10:
            M = cv2.moments(contour)
            if M["m00"] != 0:  # To avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

                # Draw the contour and the centroid on the image
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return centroids

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Parameters for thresholding, can be adjusted for tuning
    lower_thresh = 125  # Lower threshold for grayscale value
    upper_thresh = 255  # Upper threshold for grayscale value

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect centroids on the current frame
        centroids = detect_centroids(frame, lower_thresh, upper_thresh)

        # Display the frame with contours and centroids
        cv2.imshow('Centroid Detection', frame)

        # Print detected centroids
        print(f"Detected Centroids: {centroids}")

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()