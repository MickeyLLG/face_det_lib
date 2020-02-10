from abc import ABCMeta, abstractmethod


class face_detector(metaclass=ABCMeta):
    @abstractmethod
    def det_faces(self, img):  # return [[x1,y1,w1,h1],[x2,y2,...],...]
        pass


class landmark_detector(metaclass=ABCMeta): # return [[face1 landmarks],[face2 landmarks],...]
    @abstractmethod
    def det_landmarks(self, img, faces): # [[[x11,y11],[x12,y12],...],[[x21,...],...]]
        pass
