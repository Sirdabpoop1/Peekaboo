import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Webcam Test", frame)
        cv2.waitKey(1000)  # Show frame for 1 second
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Could not access webcam.")
