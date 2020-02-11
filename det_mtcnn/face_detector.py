from Interface import face_detector
import cv2
from mtcnn import MTCNN

detector = MTCNN()
confidence_thres = 0.98  # Threshold of object score when visualize the bboxes.


class mtcnn_face_detector(face_detector):
    def det_faces(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector.detect_faces(image)
        faces = []
        for det in dets:
            if det['confidence'] < confidence_thres:
                continue
            x, y, w, h = det['box']
            faces.append([x, y, w, h])
        return faces
