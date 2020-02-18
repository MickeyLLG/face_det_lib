import cv2
import sys

sys.path.append('det_cnn')
from mark_detector import MarkDetector
from Interface import landmark_detector

CNN_INPUT_SIZE = 128
mark_detector = MarkDetector()


class cnn_landmark_detector(landmark_detector):
    def det_landmarks(self, img, faces):
        landmarks=[]
        for facebox in faces:
            face_img = img[facebox[1]: facebox[1] + facebox[3], facebox[0]: facebox[0] + facebox[2], :]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks([face_img])
            marks[:, 0] *= facebox[2]
            marks[:, 1] *= facebox[3]
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            landmark = [[marks[i, 0], marks[i, 1]] for i in range(marks.shape[0])]
            landmarks.append(landmark)
        return landmarks
