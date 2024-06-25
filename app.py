import cv2
import numpy as np

# Function to preprocess images
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Function to match template with feature detection
def match_template(small_image, large_image):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(small_image, None)
    kp2, des2 = orb.detectAndCompute(large_image, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    matching_result = cv2.drawMatches(small_image, kp1, large_image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matching_result, len(matches)

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Load the small image to be matched
small_image = cv2.imread('brown.jpg')

if small_image is None:
    print("Error: Could not load the small image.")
    exit()

# Preprocess the small image
small_image_preprocessed = preprocess_image(small_image)

# Camera capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    large_image_preprocessed = preprocess_image(frame)

    # Perform template matching
    matching_result, match_count = match_template(small_image_preprocessed, large_image_preprocessed)
    print("Number of Matches:", match_count)

    # Display the result
    cv2.imshow('output', matching_result)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
