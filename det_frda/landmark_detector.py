import sys

sys.path.append('det_frda')
from utils import crop_img
from facerda import FaceRDA
from Interface import landmark_detector

model_path = "det_frda/model/frda_sim.onnx"
facerda = FaceRDA(model_path, True)
FRDA_CHEEK_MARKS = [32044, 33091, 31697, 31972, 45997, 45380, 44986, 44574, 44251, 43902,
                    43591, 43040, 42125, 22172, 19632, 18696, 17302]
FRDA_EYEBROW_MARKS = [14432, 13012, 39911, 10304, 9569, 6810, 6064, 5161, 4001, 2066]
FRDA_NOSE_MARKS = [8162, 8178, 8188, 8193, 9884, 9164, 8205, 7244, 6516]
FRDA_EYE_MARKS = [14067, 12384, 11354, 10456, 11493, 12654, 5829, 4921, 3887, 2216, 3641, 4802]
FRDA_MOUTH_MARKS = [10796, 10396, 8936, 8216, 7496, 6026, 5523, 6916, 7637, 8237, 8837, 9556,
                    10538, 9065, 8224, 7385, 5910, 7630, 8230, 8830]


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
            marks = []
            marks += [[vertices[0, i], vertices[1, i]] for i in FRDA_CHEEK_MARKS]  # Add cheek landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in FRDA_EYEBROW_MARKS]  # Add eyebrow landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in FRDA_NOSE_MARKS]  # Add nose landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in FRDA_EYE_MARKS]  # Add eye landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in FRDA_MOUTH_MARKS]  # Add mouth landmarks
            landmarks.append(marks)
        return landmarks

    # def det_landmarks(self, img, faces):
    #     landmarks = []
    #     for boxes in faces:
    #         roi_box = get_crop_box(boxes[0], boxes[1], boxes[2], boxes[3], 1.4)
    #         face, ret_roi = crop_img(img, roi_box)
    #         vertices = facerda(face, roi_box)
    #         marks = [[vertices[0, i], vertices[1, i]] for i in l_eye_vls]  # Add left eye landmarks
    #         marks += [[vertices[0, i], vertices[1, i]] for i in r_eye_vls]  # Add right eye landmarks
    #         landmarks.append(marks)
    #     return landmarks
