import cv2

# Open system camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Fixed bounding box coordinates
x, y, w, h = 150, 100, 250, 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw bounding box
    cv2.rectangle(frame, (x, y+70), (x + w, y + h), (0, 255, 0), 10)

    cv2.imshow("Live Camera with Bounding Box", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
