import dlib
import cv2
from Interface import landmark_detector

# print (os.path.abspath('.'))
predictor_path = "det_dlib/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


class dlib_landmark_detector(landmark_detector):
    def __init__(self):
        self.__predictor = predictor

    def det_landmarks(self, img, faces):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks=[]
        for face in faces:
            det = dlib.rectangle(int(face[0]), int(face[1]), int(face[0] + face[2]), int(face[1] + face[3]))
            shape = self.__predictor(img_gray, det)
            landmarks.append([[part.x, part.y] for part in shape.parts()])
        return landmarks
