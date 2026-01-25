import cv2

# Open system camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Fixed bounding box coordinates
x1, y1, x2, y2 = 150, 100, 250, 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # # Prepare label with class name and confidence
    label = f"Hello there"

    # Calculate label size and position
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 2, 4
    )

    # Draw label background
    cv2.rectangle(
        frame,
        (x1, y1 - label_height - baseline - 5),
        (x1 + label_width, y1),
        (0, 255, 0),
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1, y1 - baseline + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        4
    )



    cv2.imshow("Live Camera with Bounding Box", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
