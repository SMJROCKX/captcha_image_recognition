import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False #image is no longer writable
    results = model.process(image) #make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1.5, circle_radius=1))

def results_collection(results):
    all_landmarks = []
    for res in results.face_landmark.landmark:
        if results.face_landmark:
            value = np.array([res.x, res.y, res.z])
        else:
            value = np.zeros(468*3)
        all_landmarks.append(value)
    return all_landmarks.flatten()

data_path = os.path.join('MP_DATA')
actions = np.array(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z'])
no_sequences = 5
sequence_length = 30
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(data_path, action, str(sequence)))
        except:
            pass




cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # read feed from webcam
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

cap = cv2.videoCapture(0)
with mp_holistic.Holistic() as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                if frame_num == 0:
                    cv2.putText(image, 'Starting collection', (120, 200),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'collecting frames {} for video no. {} ', format(action, sequence),
                                (120, 200), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'collecting frames {} for video no. {} ', format(action,sequence),
                                (120, 200), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 4, cv2.LINE_AA)
                keypoints = results_collection(results)
                npy_path  = os.path.join(data_path,action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
