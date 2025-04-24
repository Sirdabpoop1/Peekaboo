import cv2
import face_recognition
import numpy as np
from db import get_faces, new_face, close_db

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video :(")
        break
    
    cv2.imshow("Face Recognition App", frame)

    key = cv2.waitKey(1)

    if key == ord(" "):
        print("Shot Taken!")

        known_names, known_encodings = get_faces()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_encodings) == 0:
            print("No faces detected in the frame!")
            continue

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)

            if True in matches:
                match_index = matches.index(True)
                recognized_name = known_names[match_index]
                print(f"Recognized: {recognized_name}")
            else:
                name = input("New face detected. Enter name: ")
                new_face(name, face_encoding)
                print(f"Saved {name} to database!")

    elif key == ord("q"):
        print("Quitting...")
        break

video.release()
cv2.destroyAllWindows()
close_db()