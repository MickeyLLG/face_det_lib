import sys

sys.path.append('det_frda')
from utils import crop_img
from facerda import FaceRDA
from Interface import landmark_detector

model_path = "det_frda/model/frda_sim.onnx"
l_eye_vls_path = 'det_frda/model/l_eye_vls.txt'
r_eye_vls_path = 'det_frda/model/r_eye_vls.txt'
l_eye_vls = [eval(v.replace('\n', '')) for v in open(l_eye_vls_path, 'r')]
r_eye_vls = [eval(v.replace('\n', '')) for v in open(r_eye_vls_path, 'r')]
facerda = FaceRDA(model_path, True)
eye_marks = [1446, 3627, 5179, 5965, 4803, 3386, 10083, 10705, 12121, 14587, 13043, 11754]
eyebrow_marks = [2066, 4001, 5161, 6064, 6810, 9569, 10304, 39911, 13012, 14432]
nose_marks = [8154, 8054, 8185, 8192, 6012, 6883, 8085, 9284, 10008]
outline_marks = [17302, 18696, 19632, 22172, 42125, 43040, 43591, 43902, 44251,
                 44574, 44986, 45380, 45997, 29160, 31697, 33091, 32044]


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
            marks = [[vertices[0, i], vertices[1, i]] for i in eye_marks]  # Add eye landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in eyebrow_marks]  # Add eyebrow landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in nose_marks]  # Add nose landmarks
            marks += [[vertices[0, i], vertices[1, i]] for i in outline_marks]  # Add outline landmarks
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
