import onnxruntime as ort
import cv2
import numpy as np
from Interface import landmark_detector


class pig_landmark_detector(landmark_detector):
    def __init__(self):
        self.session = ort.InferenceSession("./det_pig/models/keypoints.onnx", None)
        self.x0 = self.session.get_inputs()[0].name
        self.x1 = self.session.get_inputs()[1].name
        self.output = [self.session.get_outputs()[0].name]
        self.extend_range = 0.2
        self.point_num = 68

    def det_landmarks(self, img, faces):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = []
        for bbox in faces:
            bbox = np.array(bbox)
            bbox_width = bbox[2]
            bbox_height = bbox[3]
            add = int(max(bbox_width, bbox_height))
            bimg = cv2.copyMakeBorder(img, add, add, add, add,
                                      borderType=cv2.BORDER_CONSTANT, value=[123., 116., 103.])
            bbox += add
            one_edge = (1 + 2 * self.extend_range) * bbox_width
            center = [(bbox[0] + bbox_width // 2), (bbox[1] + bbox_height // 2)]

            bbox[0] = center[0] - one_edge // 2
            bbox[1] = center[1] - one_edge // 2
            bbox[2] = center[0] + one_edge // 2
            bbox[3] = center[1] + one_edge // 2
            bbox = bbox.astype(np.int32)

            crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            cv2.imshow('1', crop_image)
            face_input = cv2.resize(crop_image, (160, 160))
            face_input = np.expand_dims(face_input, 0).astype(np.float32)
            lmks = self.session.run(self.output,
                                    {self.x0: face_input, self.x1: np.array(False)})[0][:, :2 * self.point_num]
            lmks = lmks.reshape(-1, 2)

            lmks[:, 0] *= crop_image.shape[1]
            lmks[:, 1] *= crop_image.shape[0]
            lmks[:, 0] = lmks[:, 0] + bbox[0] - add
            lmks[:, 1] = lmks[:, 1] + bbox[1] - add

            landmarks.append(lmks.tolist())
        return landmarks
