import cv2

def capture_and_save_image():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    # Capture one frame
    ret, frame = cap.read()

    # Check if the frame is correctly captured
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    # Save the captured image
    cv2.imwrite('driving-with_phone.jpg', frame)

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_image()

print("Finished")