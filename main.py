import cv2
import numpy as np
import requests



def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def match_template(small_image, large_image, method):
    result = cv2.matchTemplate(small_image, large_image, method)
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)
    score = mn
    return score, mnLoc


cap = cv2.VideoCapture(0)
small_image = cv2.imread("driving-with_phone.jpg")

if small_image is None:
    print("Error: Could not load the small image.")
    exit()


small_image_preprocessed = preprocess_image(small_image)
method = cv2.TM_CCOEFF_NORMED


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    large_image_preprocessed = preprocess_image(frame)


    score, mnLoc = match_template(
        small_image_preprocessed, large_image_preprocessed, method
    )
    score = abs(score)
    score = score * 1000
    print("Matching Score:", score)

    if(score > 17):
        try:
            x = requests.get("https://sms-api-hbr7.onrender.com/sms?plate=T266AFV")
            print(x.json())
        except Exception as e:
            print(e)


    MPx, MPy = mnLoc
    trows, tcols = small_image.shape[:2]
    cv2.rectangle(frame, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)


    cv2.imshow("output", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
