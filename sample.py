import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error loading Haar cascade!")
else:
    print("Haar cascade loaded successfully!")
