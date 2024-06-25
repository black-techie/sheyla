import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Function to detect hands
def detect_hands(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    hand_landmarks = result.multi_hand_landmarks
    return hand_landmarks

# Function to check if phone is near hand
def is_phone_near_hand(phone_bbox, hand_landmarks):
    x1, y1, x2, y2 = phone_bbox
    for hand_landmark in hand_landmarks:
        for lm in hand_landmark.landmark:
            if x1 < lm.x < x2 and y1 < lm.y < y2:
                return True
    return False

# Load video
cap = cv2.VideoCapture('path_to_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect phones
    phones = detect_phone(rgb_frame)
    
    # Detect hands
    hand_landmarks = detect_hands(frame)
    
    # Check if any phone is near hands
    for phone in phones:
        if hand_landmarks and is_phone_near_hand(phone, hand_landmarks):
            x1, y1, x2, y2 = map(int, phone)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            x1, y1, x2, y2 = map(int, phone)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
