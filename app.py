import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
from db import get_faces, new_face, close_db

current_command = ""
listen_flag = True
processing_face = False

speech_queue = queue.Queue()
command_queue = queue.Queue()

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.1)

def speech_loop():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

speech_thread = threading.Thread(target=speech_loop, daemon=True)
speech_thread.start()

def speak(text):
    speech_queue.put(text)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.2
recognizer.dynamic_energy_threshold = True
recognizer.dynamic_energy_adjustment_damping = 0.15


video = cv2.VideoCapture(0)

def listen_once(prompt = ""):
    mic = sr.Microphone(device_index=2)
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        if prompt:
            speak(prompt)
        try:
            audio = recognizer.listen(source, timeout = 3, phrase_time_limit = 5)
            command = recognizer.recognize_google(audio)
            print(f"Heard: {command}")
            return command.lower()
        except sr.WaitTimeoutError:
            if prompt:
                speak("Listening timed out")
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            speak("Speech service unavailable")
            return None

        except Exception as e:
            print(f"Microphone error: {e}")
            return None
    
def listen_command_loop(command_queue):
    global current_command

    while listen_flag:
        if not processing_face:
            command = listen_once()
            if command:
                command_queue.put(command)
    

def recognize_faces(frame):
    global processing_face
    processing_face = True
    known_names, known_encodings = get_faces()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    names_in_frame = []

    if len(face_encodings) == 0:
        speak("No faces detected in the frame!")
        processing_face = False
        return [], []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance = 0.4)

        if True in matches:
            match_index = matches.index(True)
            recognized_name = known_names[match_index]
            speak(f"Recognized: {recognized_name}")
            names_in_frame.append(recognized_name)
        else:
            speak("I don't know this person.")
            confirmed = False
            attempts = 0   
            while not confirmed:
                name_audio = listen_once("Please say their name")
                time.sleep(3)
                if name_audio:
                    confirm = listen_once(f"Did you say {name_audio}?")
                    time.sleep(3)
                    if confirm and "yes" in confirm:
                        confirmed = True
                    else:
                        speak("Let's try again")
                time.sleep(3)
                attempts += 1
            if name_audio and confirmed:
                new_face(name_audio, face_encoding)
                speak(f"Saved {name_audio}")
                names_in_frame.append(name_audio)
            else:
                speak("Sorry, I didn't catch the name.")
                names_in_frame.append("Unknown")

    processing_face = False
    return face_locations, names_in_frame

listener_thread = threading.Thread(target=listen_command_loop, args = (command_queue,), daemon=True)
listener_thread.start()

try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed Video")
            break

        small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        names_in_frame = []

        if not command_queue.empty():
            current_command = command_queue.get()
        
        if "who is this" in current_command:
            speak("Looking now")
            face_locations, names_in_frame = recognize_faces(rgb_small_frame)
            current_command = ""
        elif "quit" in current_command or "stop" in current_command or 0xFF == ord('q'):
            speak("Shutting down")
            break
        
        if not names_in_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
        
        for (top, right, bottom, left), name in zip(face_locations, names_in_frame or ["Unknown"] * len(face_locations)):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)
        
except KeyboardInterrupt:
    pass

finally:
    listen_flag = False
    time.sleep(0.1)
    listener_thread.join(timeout = 2)
    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    close_db()
    time.sleep(0.2)
    speech_queue.put(None)
    speech_thread.join(timeout = 2)
