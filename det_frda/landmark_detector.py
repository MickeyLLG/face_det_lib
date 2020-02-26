import sys

sys.path.append('det_frda')
from utils import crop_img
from facerda import FaceRDA
from Interface import landmark_detector

model_path = "det_frda/model/frda_sim.onnx"
facerda = FaceRDA(model_path, True)
mark_ls = [1446, 3626, 5178, 5965, 4803, 3257, 10083, 10705, 12121, 14587, 12915, 11753]


def get_crop_box(x1, y1, w, h, scale=1.1):
    size = int(max([w, h]) * scale)
    cx = x1 + w // 2
    cy = int(y1 + h // 2 + h * 0.1)
    # cy = int(y1 + h // 2 + h * 0)

    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    return x1, y1, x2, y2


class frda_landmark_detector(landmark_detector):

    def det_landmarks(self, img, faces):
        landmarks = []
        for boxes in faces:
            roi_box = get_crop_box(boxes[0], boxes[1], boxes[2], boxes[3], 1.4)
            face, ret_roi = crop_img(img, roi_box)
            vertices = facerda(face, roi_box)
            marks=[[vertices[0, i], vertices[1, i]] for i in mark_ls]
            landmarks.append(marks)
        return landmarks
