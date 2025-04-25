import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import threading
import queue
from db import get_faces, new_face, close_db

current_command = ""
listen_flag = True

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

recognizer = sr.Recognizer()
mic = sr.Microphone()

video = cv2.VideoCapture(0)

speech_queue = queue.Queue()

def recognize_faces(frame):

    known_names, known_encodings = get_faces()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) == 0:
        print("No faces detected in the frame!")

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance = 0.4)

        if True in matches:
            match_index = matches.index(True)
            recognized_name = known_names[match_index]
            print(f"Recognized: {recognized_name}")
        else:
            speech_queue.put("I don't know this person. Please say their name after the beep.")
            name_audio = listen_command()

            if name_audio:
                new_face(name_audio, face_encoding)
                speech_queue.put(f"Saved {name_audio}")
            else:
                speech_queue.put("Sorry, I didn't catch the name.")

def listen_command():
    global current_command
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            speech_queue.put("Listening for command")
            try:
                audio = recognizer.listen(source, timeout = 5, phrase_time_limit = 5)
                command = recognizer.recognize_google(audio)
                current_command = command.lower()
                print(f"Command: {current_command}")
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                speech_queue.put("I didn't catch that")
            except sr.RequestError:
                speech_queue.put("Speech unavailable")

def listen_in_background():
    while listen_flag:
        listen_command()

listener_thread = threading.Thread(target=listen_in_background, daemon=True)
listener_thread.start()    

def handle_speech():
    while not speech_queue.empty():
        message = speech_queue.get()
        speak(message)

try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed Video")
            break
        cv2.imshow("Camera Feed", frame)
        if "Who is this" in current_command:
            speech_queue.put("Looking now")
            recognize_faces(frame)
            current_command = ""
        elif "quit" in current_command or "stop" in current_command:
            speech_queue.put("Shutting down")
            break
        
        handle_speech()     
        cv2.waitKey(1)
        
except KeyboardInterrupt:
    pass

finally:
    listen_flag = False
    listener_thread.join()
    video.release()
    cv2.destroyAllWindows()
    close_db()