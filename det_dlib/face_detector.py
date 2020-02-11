from Interface import face_detector
import dlib
import cv2


class dlib_face_detector(face_detector):
    def __init__(self):
        self.__detector = dlib.get_frontal_face_detector()

    def det_faces(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.__detector(img_gray, 0)
        faces = []
        for det in dets:
            l = det.left()
            t = det.top()
            r = det.right()
            b = det.bottom()
            faces.append([l, t, r - l, b - t])
        return faces


if __name__ == '__main__':
    d = dlib_face_detector()
