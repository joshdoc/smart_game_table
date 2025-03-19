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

    # Configuration option for tuning the spacing threshold between centroids (in pixels)
    DOT_SPACING_THRESHOLD = 75  # Adjust this value as needed

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
        
        # Prepare to store centroids for game pieces
        game_piece_centroids = []

        # Loop over the contours to detect and draw centroids
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out very small contours (adjust the threshold as needed)
            if area < 15:
                continue

            # Detected contour is a fingerprint
            if area > 45:
                # Calculate moments for each contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, "touch", (cX - 25, cY - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Detected contour is part of a game piece
            else:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    game_piece_centroids.append((cX, cY))

        # Cluster the game piece centroids based on their proximity
        clusters = []
        for pt in game_piece_centroids:
            added = False
            for cluster in clusters:
                # If any point in the cluster is within the threshold distance, add this point to that cluster
                if any(np.linalg.norm(np.array(pt) - np.array(existing_pt)) < DOT_SPACING_THRESHOLD for existing_pt in cluster):
                    cluster.append(pt)
                    added = True
                    break
            if not added:
                clusters.append([pt])

        # For each cluster, decide if it is a "pawn" (one dot) or a "rook" (four dots)
        for cluster in clusters:
            # Calculate the average position of the centroids in the cluster
            avg_x = int(sum(p[0] for p in cluster) / len(cluster))
            avg_y = int(sum(p[1] for p in cluster) / len(cluster))
            if len(cluster) == 1:
                cv2.putText(frame, "pawn", (avg_x - 25, avg_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif len(cluster) == 2:
                cv2.putText(frame, "knight", (avg_x - 25, avg_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    
            elif len(cluster) == 3:
                cv2.putText(frame, "bishop", (avg_x - 25, avg_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif len(cluster) == 4:
                cv2.putText(frame, "rook", (avg_x - 25, avg_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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
